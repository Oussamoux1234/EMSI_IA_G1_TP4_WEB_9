package ma.emsi.essalmani.tp4webessalmani.llm;



import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;

/**
 * Classe métier qui gère la connexion à Gemini via LangChain4j.
 */
@Dependent
public class LlmClient implements Serializable {

    private String systemRole;
    private Assistant assistant;
    private ChatMemory chatMemory;

    /**
     * Interface définissant l’interaction "chat".
     * LangChain4j génère automatiquement l’implémentation.
     */
    public interface Assistant {
        String chat(String prompt);
    }

    public LlmClient() {

        String apiKey = System.getenv("GEMINI_KEY");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .build();
    }

    /**
     * Setter appelé lorsque l'utilisateur choisit le rôle système.
     * On vide la mémoire et on ajoute un message système.
     */

    public void setSystemRole(String role) {
        this.systemRole = role;
        chatMemory.clear();
        chatMemory.add(SystemMessage.from(role));
    }

    /**
     * Envoie la question au LLM et retourne la réponse.
     */
    public String ask(String question) {
        return assistant.chat(question);
    }

}
