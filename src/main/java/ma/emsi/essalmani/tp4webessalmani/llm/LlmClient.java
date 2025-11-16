package ma.emsi.essalmani.tp4webessalmani.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Named;

import java.io.Serializable;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * LlmClient - implémentation Test 5 (RAG avec 2 PDFs + Tavily)
 */
@Dependent
public class LlmClient implements Serializable {

    private static final long serialVersionUID = 1L;

    // Public interface used elsewhere in your app implicitly (AiServices uses its own Assistant)


    private String systemRole;
    private ChatMemory chatMemory;

    // LangChain models / assistant
    private ChatModel chatModel;

    // LangChain assistant (with retrieval augumentor) — created in init
    private Assistant assistant;

    // Embedding model etc (kept as fields in case you want to reuse)
    private AllMiniLmL6V2EmbeddingModel embeddingModel;
    private RetrievalAugmentor retrievalAugmentor;

    public LlmClient() {
        // empty constructor for CDI
    }

    @PostConstruct
    public void init() {
        try {
            configureLogger();

            String geminiKey = System.getenv("GEMINI_KEY");
            String tavilyKey = System.getenv("TAVILY_KEY");

            if (geminiKey == null || geminiKey.isBlank()) {
                throw new IllegalStateException("GEMINI_KEY not set in environment variables");
            }
            if (tavilyKey == null || tavilyKey.isBlank()) {
                throw new IllegalStateException("TAVILY_KEY not set in environment variables");
            }

            // ---------------- Chat model ----------------
            chatModel = GoogleAiGeminiChatModel.builder()
                    .apiKey(geminiKey)
                    .modelName("gemini-2.5-flash")
                    .temperature(0.2)
                    .timeout(Duration.ofSeconds(60))
                    .logRequestsAndResponses(true)
                    .build();

            // ---------------- Embeddings + ingestion for two PDFs ----------------
            embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            // load rag.pdf
            EmbeddingStore<TextSegment> storeRag = createEmbeddingStoreFromResource("/rag.pdf");

            // load rse.pdf
            EmbeddingStore<TextSegment> storeRse = createEmbeddingStoreFromResource("/RSE.pdf");

            // create retrievers for the two pdf stores
            ContentRetriever retrieverPdf1 = EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(storeRag)
                    .embeddingModel(embeddingModel)
                    .maxResults(3)
                    .minScore(0.5)
                    .build();

            ContentRetriever retrieverPdf2 = EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(storeRse)
                    .embeddingModel(embeddingModel)
                    .maxResults(3)
                    .minScore(0.5)
                    .build();

            // ---------------- Web search (Tavily) ----------------
            TavilyWebSearchEngine webEngine = TavilyWebSearchEngine.builder()
                    .apiKey(tavilyKey)
                    .build();

            ContentRetriever retrieverWeb = WebSearchContentRetriever.builder()
                    .webSearchEngine(webEngine)
                    .maxResults(3)
                    .build();

            // ---------------- Build router + augmentor ----------------
            DefaultQueryRouter router = new DefaultQueryRouter(retrieverPdf1, retrieverPdf2, retrieverWeb);

            this.retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                    .queryRouter(router)
                    .build();

            // ---------------- Chat memory and assistant ----------------
            this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

            this.assistant = AiServices.builder(Assistant.class)
                    .chatModel(chatModel)
                    .chatMemory(chatMemory)
                    .retrievalAugmentor(this.retrievalAugmentor)
                    .build();

            // init done
            Logger.getLogger(LlmClient.class.getName()).info("LlmClient initialized (Test5 RAG + Tavily)");

        } catch (RuntimeException rte) {
            // rethrow runtime exceptions as-is
            throw rte;
        } catch (Exception e) {
            // wrap checked exceptions to avoid @PostConstruct checked exception problem
            throw new RuntimeException("Erreur lors de l'initialisation du LlmClient (Test5)", e);
        }
    }

    /**
     * Set the system role (keeps same behavior as your TP2).
     * Clears chat memory and add system message.
     */
    public void setSystemRole(String role) {
        this.systemRole = role;
        if (this.chatMemory != null) {
            this.chatMemory.clear();
            // add as first system message (LangChain4j SystemMessage helper)
            this.chatMemory.add(dev.langchain4j.data.message.SystemMessage.from(role));
        }
    }

    /**
     * Main public method used by the web app.
     * This method uses the assistant configured with retrievalAugmentor (Test5).
     */
    public String ask(String prompt) {
        try {
            if (this.assistant == null) {
                throw new IllegalStateException("Assistant not initialized");
            }
            // delegate to LangChain4j assistant (it will use the augmentor/router automatically)
            return this.assistant.chat(prompt);
        } catch (Exception e) {
            // return a user-friendly message (avoid leaking stacktraces to UI)
            e.printStackTrace();
            return "Impossible de contacter Gemini : " + e.getMessage();
        }
    }

    // ---------------- Utility methods ----------------

    private EmbeddingStore<TextSegment> createEmbeddingStoreFromResource(String resourcePath) throws Exception {
        URL url = LlmClient.class.getResource(resourcePath);
        if (url == null) {
            throw new IllegalArgumentException("Ressource introuvable: " + resourcePath + ". Place the file in src/main/resources/");
        }
        Path path = Paths.get(url.toURI());

        Document document = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);
        return store;
    }

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler console = new ConsoleHandler();
        console.setLevel(Level.FINE);
        logger.addHandler(console);
    }
}
