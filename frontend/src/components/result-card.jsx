import {
  ChevronDown,
  ChevronUp,
  FileText,
  Flame,
  Loader2,
  Sparkles,
} from "lucide-react";
import { useState } from "react";
import { apiService } from "../services/api";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card } from "./ui/card";

export function ResultCard({ result, query, allResults }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhancedText, setEnhancedText] = useState(null);
  const [showEnhanced, setShowEnhanced] = useState(false);
  const [enhancementError, setEnhancementError] = useState(null);

  const getSimilarityColor = (similarity) => {
    const value = parseFloat(similarity);
    if (value >= 0.8) return "text-success";
    if (value >= 0.6) return "text-accent";
    return "text-muted-foreground";
  };

  const handleReadMore = () => {
    setIsExpanded(!isExpanded);
  };

  const handleEnhance = async () => {
    setIsEnhancing(true);
    setEnhancementError(null);

    try {
      console.log("🔮 Enhancing chunk #", result.id);
      console.log("📝 Query:", query);
      console.log("📊 Result index:", result.id - 1);
      console.log("📦 All results:", allResults);

      // Call the enhance API with the current result
      const response = await apiService.enhanceRemaining({
        query: query,
        results: allResults,
        indices: [result.id - 1],
      });

      console.log("✅ Full Enhancement response:", response);

      // The backend returns: { results: [...], enhanced: int, ... }
      // Each result in results has: enhanced_text, relevance, enhancement_status
      if (response.results && response.results.length > 0) {
        const enhancedResult = response.results[result.id - 1];
        console.log(
          "📝 Enhanced result at index:",
          result.id - 1,
          enhancedResult
        );

        if (enhancedResult && enhancedResult.enhanced_text) {
          setEnhancedText(enhancedResult.enhanced_text);
          setShowEnhanced(true);
          console.log("✅ Successfully set enhanced text");
        } else {
          throw new Error("Enhanced text not found in response");
        }
      } else {
        throw new Error("No results returned from enhancement API");
      }
    } catch (error) {
      console.error("❌ Enhancement failed:", error);
      setEnhancementError(
        error.message || "Failed to enhance text. Please try again."
      );
      setEnhancedText(null);
    } finally {
      setIsEnhancing(false);
    }
  };

  const toggleView = () => {
    setShowEnhanced(!showEnhanced);
  };

  const getDisplayText = () => {
    if (showEnhanced && enhancedText) {
      return enhancedText;
    }
    return isExpanded ? result.fullContent || result.preview : result.preview;
  };

  return (
    <Card className="p-5 space-y-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-accent shrink-0" />
          <h3 className="font-semibold text-lg">Result #{result.id}</h3>
          {showEnhanced && enhancedText && (
            <Badge variant="default" className="bg-accent gap-1">
              <Sparkles className="w-3 h-3" />
              Enhanced
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-xs text-muted-foreground">Similarity:</span>
          <span
            className={`font-bold ${getSimilarityColor(result.similarity)}`}
          >
            {result.similarity}
          </span>
          {parseFloat(result.similarity) >= 0.8 && (
            <Flame className="w-4 h-4 text-destructive" />
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-sm">
        <Badge variant="secondary" className="gap-1">
          {result.category}
        </Badge>
        <Badge variant="outline" className="gap-1">
          {result.url}
        </Badge>
        <Badge variant="outline" className="gap-1">
          Chunk {result.chunk}
        </Badge>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">
          {showEnhanced && enhancedText
            ? "✨ AI-Enhanced Summary:"
            : "📝 Content Preview:"}
        </p>
        <p
          className={`text-sm leading-relaxed text-foreground/90 ${
            isExpanded || (showEnhanced && enhancedText) ? "" : "line-clamp-3"
          }`}
        >
          {getDisplayText()}
        </p>
        {enhancementError && (
          <p className="text-sm text-destructive">⚠️ {enhancementError}</p>
        )}
      </div>

      <div className="flex flex-wrap gap-2 pt-2">
        <Button
          variant="default"
          size="sm"
          className="bg-accent hover:bg-accent/90"
          onClick={handleReadMore}
          disabled={isEnhancing}
        >
          {isExpanded ? (
            <>
              <ChevronUp className="w-3 h-3 mr-1" />
              Show Less
            </>
          ) : (
            <>
              <ChevronDown className="w-3 h-3 mr-1" />
              Read More
            </>
          )}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={handleEnhance}
          disabled={isEnhancing || !query || !allResults}
        >
          {isEnhancing ? (
            <>
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
              Enhancing...
            </>
          ) : (
            <>
              <Sparkles className="w-3 h-3 mr-1" />
              AI Enhance
            </>
          )}
        </Button>

        {enhancedText && (
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleView}
            disabled={isEnhancing}
          >
            {showEnhanced ? "Show Original" : "Show Enhanced"}
          </Button>
        )}
      </div>
    </Card>
  );
}
