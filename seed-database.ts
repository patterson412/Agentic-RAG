import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { ClientSecretCredential } from "@azure/identity";
import axios from "axios";
import "dotenv/config";
import pdf from 'pdf-parse';
import mammoth from 'mammoth';

// Initialize MongoDB client
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

// Azure AD and SharePoint configuration
const TENANT_ID = process.env.TENANT_ID as string;
const CLIENT_ID = process.env.CLIENT_ID as string;
const CLIENT_SECRET = process.env.CLIENT_SECRET as string;
const SITE_ID = process.env.SITE_ID as string;

class SharePointService {
  private credential: ClientSecretCredential;
  private accessToken: string | null = null;

  constructor() {
    this.credential = new ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET);
  }

  // Retrieve access token from Azure AD
  async getAccessToken(): Promise<string> {
    if (!this.accessToken) {
      const response = await this.credential.getToken("https://graph.microsoft.com/.default");
      if (!response?.token) {
        throw new Error("Failed to get access token from Azure AD");
      }
      this.accessToken = response.token;
    }
    return this.accessToken;
  }

  // List documents in the default document library
  async listDocuments() {
    const token = await this.getAccessToken();
    const drivesEndpoint = `https://graph.microsoft.com/v1.0/sites/${SITE_ID}/drives`;

    const drivesResponse = await axios.get(drivesEndpoint, {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (!drivesResponse.data?.value) {
      throw new Error("No drives found in SharePoint response");
    }

    // Assuming you want to list documents from the first drive
    const driveId = drivesResponse.data.value[0].id;
    const documentsEndpoint = `https://graph.microsoft.com/v1.0/drives/${driveId}/root/children`;

    const documentsResponse = await axios.get(documentsEndpoint, {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (!documentsResponse.data?.value) {
      throw new Error("No documents found in SharePoint response");
    }

    return documentsResponse.data.value.map((item: any) => ({
      id: item.id,
      name: item.name || 'Untitled',
      webUrl: item.webUrl,
      lastModified: item.lastModifiedDateTime,
    }));
  }

  // Extract text from PDF files
  private async extractPdfText(arrayBuffer: ArrayBuffer): Promise<string> {
    const data = await pdf(Buffer.from(arrayBuffer));
    return data.text;
  }

  // Extract text from DOCX files
  private async extractDocxText(arrayBuffer: ArrayBuffer): Promise<string> {
    const result = await mammoth.extractRawText({ arrayBuffer });
    return result.value;
  }

  // Retrieve and process document content based on file type
  async getDocumentContent(documentId: string): Promise<string> {
    const token = await this.getAccessToken();
    const drivesEndpoint = `https://graph.microsoft.com/v1.0/sites/${SITE_ID}/drives`;
    const drivesResponse = await axios.get(drivesEndpoint, {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (!drivesResponse.data?.value) {
      throw new Error("No drives found in SharePoint response");
    }

    const driveId = drivesResponse.data.value[0].id;
    const endpoint = `https://graph.microsoft.com/v1.0/drives/${driveId}/items/${documentId}/content`;

    const response = await axios.get(endpoint, {
      headers: { Authorization: `Bearer ${token}` },
      responseType: "arraybuffer", // Ensure the response is an ArrayBuffer
    });

    const contentType = response.headers['content-type'];

    try {
      if (contentType === 'application/pdf') {
        return this.extractPdfText(response.data);
      } else if (contentType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        return this.extractDocxText(response.data);
      } else {
        // Default text decoding for other types
        const decoder = new TextDecoder('utf-8');
        return decoder.decode(response.data);
      }
    } catch (error) {
      if (error instanceof Error) {
        console.error(`Text extraction failed: ${error.message}`);
      }
      throw error;
    }
  }
}

async function seedDatabase(): Promise<void> {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("Successfully connected to MongoDB!");

    const db = client.db("knowledge_base");
    const collection = db.collection("documents");

    // Clear existing documents
    await collection.deleteMany({});
    console.log("Cleared existing documents");

    const sharePointService = new SharePointService();
    const documents = await sharePointService.listDocuments();
    console.log(`Found ${documents.length} documents in SharePoint`);

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const allDocumentsWithMetadata = [];

    for (const doc of documents) {
      console.log(`Processing: ${doc.name}`);

      const content = await sharePointService.getDocumentContent(doc.id);
      const chunks = await textSplitter.createDocuments([content]);

      const documentsWithMetadata = chunks.map((chunk, index) => ({
        pageContent: chunk.pageContent,
        metadata: {
          documentId: doc.id,
          documentName: doc.name,
          webUrl: doc.webUrl,
          lastModified: doc.lastModified,
          chunkIndex: index,
        },
      }));

      allDocumentsWithMetadata.push(...documentsWithMetadata);
      console.log(`Processed: ${doc.name}`);
    }

    if (allDocumentsWithMetadata.length === 0) {
      throw new Error('allDocumentsWithMetadata array is empty to upload to database');
    }
    
    // Batch insert all documents at once
    await MongoDBAtlasVectorSearch.fromDocuments(
      allDocumentsWithMetadata,
      new OpenAIEmbeddings(),
      {
        collection,
        indexName: "vector_index",
        textKey: "embedding_text",
        embeddingKey: "embedding",
      }
    );

    console.log(`Successfully inserted ${allDocumentsWithMetadata.length} chunks into MongoDB`);
    console.log("Database seeding completed");
  } catch (error) {
    console.error("Error seeding database:", error);
    process.exit(1);
  } finally {
    await client.close();
  }
}

seedDatabase().catch(console.error);
