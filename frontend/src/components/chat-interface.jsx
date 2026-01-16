import { Loader2, Send, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { apiService } from "../services/api";
import { ChatMessage } from "./chat-message";
import { ResultCard } from "./result-card";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

export function ChatInterface({
  searchMode,
  numResults,
  categories,
  keywords,
  semanticWeight,
  messages,
  setMessages,
  currentChatId,
  prefilledQuery,
  setPrefilledQuery,
}) {
  const [input, setInput] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const hasProcessedQuery = useRef(false);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (prefilledQuery && prefilledQuery.trim() && !hasProcessedQuery.current) {
      hasProcessedQuery.current = true;

      requestAnimationFrame(() => {
        setInput(prefilledQuery);
        setPrefilledQuery("");

        setTimeout(() => {
          if (textareaRef.current) {
            textareaRef.current.focus();
          }
        }, 100);
      });
    }

    if (!prefilledQuery) {
      hasProcessedQuery.current = false;
    }
  }, [prefilledQuery, setPrefilledQuery]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isSearching) return;

    const userMessage = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const queryText = input;
    setInput("");
    setIsSearching(true);

    try {
      console.log("Search Mode:", searchMode);
      console.log("Query:", queryText);
      console.log("Num Results:", numResults);
      console.log("Categories:", categories);

      let data;

      if (searchMode === "semantic") {
        console.log("Calling SEMANTIC search endpoint");
        data = await apiService.semanticSearch({
          query: queryText,
          n_results: numResults,
          filter_category: categories.includes("all") ? null : categories[0],
        });
      } else {
        console.log("Calling HYBRID search endpoint");
        const keywordList = keywords.split(" ").filter((k) => k.trim());
        console.log("Keywords:", keywordList);
        console.log("Weights:", semanticWeight, "/", 100 - semanticWeight);

        data = await apiService.hybridSearch({
          query: queryText,
          keywords: keywordList,
          n_results: numResults,
          filter_category: categories.includes("all") ? null : categories[0],
          semantic_weight: semanticWeight / 100,
          keyword_weight: (100 - semanticWeight) / 100,
        });
      }

      console.log("Backend Response:", data);

      const botMessage = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: `Search completed!\n\nConfiguration:\n- Mode: ${
          searchMode === "semantic" ? "Semantic Search" : "Hybrid Search"
        }\n- Results: ${data.count} chunks found\n- Filter: ${
          categories.includes("all") ? "All Categories" : categories[0]
        }${
          searchMode === "hybrid" && keywords.trim()
            ? `\n- Keywords: ${keywords}\n- Weights: ${semanticWeight}% semantic, ${
                100 - semanticWeight
              }% keyword`
            : ""
        }${
          data.cached ? "\n- Cached result (faster!)" : "\n- Fresh result"
        }\n\nSearch completed successfully`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);

      const formattedResults = data.results.map((result, index) => {
        const similarity =
          searchMode === "hybrid" ? result.hybrid_score : result.similarity;

        return {
          id: index + 1,
          similarity:
            typeof similarity === "number" ? similarity.toFixed(3) : similarity,
          category: result.metadata.category,
          url: result.metadata.domain || result.metadata.url.substring(0, 30),
          chunk: `${result.metadata.chunk_index}/${result.metadata.total_chunks}`,
          preview: result.text.substring(0, 300) + "...",
          fullContent: result.text,
        };
      });

      console.log("Formatted Results:", formattedResults);

      const resultsMessage = {
        id: (Date.now() + 2).toString(),
        type: "results",
        content: "",
        timestamp: new Date(),
        results: formattedResults,
        query: queryText,
        rawResults: data.results,
      };
      setMessages((prev) => [...prev, resultsMessage]);
    } catch (error) {
      console.error("Search failed:", error);

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        type: "system",
        content: `Search failed\n\nError: ${error.message}\n\nTroubleshooting:\n- Check if backend is running\n- Check browser console for details\n- Try a different query\n- Check your network connection`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] space-y-4 text-center">
              <div className="w-16 h-16 rounded-full bg-accent/10 flex items-center justify-center">
                <Sparkles className="w-8 h-8 text-accent" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-semibold">Ready to search</h3>
                <p className="text-muted-foreground max-w-md">
                  Ask me anything about machine learning, wildlife conservation,
                  nuclear testing, or TensorFlow optimization!
                </p>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <div key={message.id}>
                  {message.type === "results" ? (
                    <div className="space-y-3">
                      {message.results?.map((result) => (
                        <ResultCard
                          key={result.id}
                          result={result}
                          query={message.query}
                          allResults={message.rawResults}
                        />
                      ))}
                    </div>
                  ) : (
                    <ChatMessage message={message} />
                  )}
                </div>
              ))}
              {isSearching && (
                <div className="flex items-center gap-2 text-muted-foreground p-4">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Processing your query...</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </div>

      <div className="border-t border-border bg-background p-4 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="space-y-3">
            <div className="relative">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your query here... (Shift+Enter for new line)"
                className="min-h-[100px] pr-12 resize-none"
                disabled={isSearching}
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isSearching}
                className="absolute bottom-2 right-2 bg-accent hover:bg-accent/90"
              >
                {isSearching ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Try: "How does TensorFlow optimize graphs?" or use /help for
              commands
            </p>
          </form>
        </div>
      </div>
    </div>
  );
}
