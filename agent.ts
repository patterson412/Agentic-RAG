import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { z } from "zod";
import "dotenv/config";

export async function callAgent(client: MongoClient, query: string, thread_id: string) {
  // Define the MongoDB database and collection
  const dbName = "knowledge_base";
  const db = client.db(dbName);
  const collection = db.collection("documents");

  // Define the graph state
  const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y),
    }),
  });

  // Define the tools for the agent to use
  const documentSearchTool = tool(
    async ({ query, n = 10 }) => {
      console.log("Document search tool called");

      const dbConfig = {
        collection: collection,
        indexName: "vector_index",
        textKey: "embedding_text",
        embeddingKey: "embedding",
      };

      // Initialize vector store
      const vectorStore = new MongoDBAtlasVectorSearch(
        new OpenAIEmbeddings(),
        dbConfig
      );

      const result = await vectorStore.similaritySearchWithScore(query, n);
      return JSON.stringify(result);
    },
    {
      name: "document_search",
      description: "Searches through Microsoft SharePoint documents to find relevant information",
      schema: z.object({
        query: z.string().describe("The search query"),
        n: z
          .number()
          .optional()
          .default(10)
          .describe("Number of results to return"),
      }),
    }
  );

  const tools = [documentSearchTool];

  const toolNode = new ToolNode<typeof GraphState.State>(tools);

  const model = new ChatAnthropic({
    model: "claude-3-5-sonnet-20241022",
    temperature: 0,
  }).bindTools(tools);

  // Define the function that determines whether to continue or not
  function shouldContinue(state: typeof GraphState.State) {
    const messages = state.messages;
    const lastMessage = messages[messages.length - 1] as AIMessage;

    if (lastMessage.tool_calls?.length) {
      return "tools";
    }
    return "__end__";
  }

  // Define the function that calls the model
  async function callModel(state: typeof GraphState.State) {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a knowledgeable AI assistant with access to Microsoft SharePoint documents. Your role is to:

1. Search through documents using the document_search tool to find relevant information
2. Analyze and synthesize information from multiple documents when necessary
3. Provide accurate, well-structured answers based on the document content
4. Cite specific documents or sections when providing information
5. Admit when you cannot find relevant information in the documents
6. Always prefix your final answer with "FINAL ANSWER" when you have found the complete information

When searching:
- Use specific, targeted search queries
- Consider trying multiple searches with different terms if initial results aren't sufficient
- Look for the most recent and relevant documents

You have access to the following tools: {tool_names}

{system_message}
Current time: {time}`,
      ],
      new MessagesPlaceholder("messages"),
    ]);

    const formattedPrompt = await prompt.formatMessages({
      system_message: "You are a knowledgeable assistant with access to documents stored in Microsoft SharePoint",
      time: new Date().toISOString(),
      tool_names: tools.map((tool) => tool.name).join(", "),
      messages: state.messages,
    });

    const result = await model.invoke(formattedPrompt);

    return { messages: [result] };
  }

  // Define the graph
  const workflow = new StateGraph(GraphState)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .addEdge("tools", "agent");

  // Initialize the MongoDB memory
  const checkpointer = new MongoDBSaver({ client, dbName });

  const app = workflow.compile({ checkpointer });

  const finalState = await app.invoke(
    {
      messages: [new HumanMessage(query)],
    },
    { recursionLimit: 15, configurable: { thread_id: thread_id } }
  );

  console.log(finalState.messages[finalState.messages.length - 1].content);

  return finalState.messages[finalState.messages.length - 1].content;
}