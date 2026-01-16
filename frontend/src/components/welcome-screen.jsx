import { Bot, Search, Sparkles } from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";

export function WelcomeScreen({ onQuickStart, onConfigure }) {
  const examples = [
    "How did Cold War politics lead to environmental health problems?",
    "Explain TensorFlow graph optimization techniques",
    "What are Dian Fossey's contributions to gorilla conservation?",
    "Describe the environmental impacts of nuclear testing",
  ];

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-background via-background to-accent/5">
      <div className="max-w-4xl w-full space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-primary/10 mb-4">
            <Bot className="w-10 h-10 text-primary" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-primary">
            Intelligent Content Retrieval System
          </h1>
          <p className="text-xl text-muted-foreground">
            Semantic Search Powered by AI
          </p>
        </div>

        {/* Welcome message */}
        <Card className="p-8 space-y-6 bg-card/50 backdrop-blur-sm border-2">
          <div className="space-y-4">
            <p className="text-lg leading-relaxed">
              Hi! I'm your SMART search assistant. I can help you find
              information from:
            </p>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <Sparkles className="w-5 h-5 text-accent mt-0.5 shrink-0" />
                <span>Machine Learning & AI concepts</span>
              </li>
              <li className="flex items-start gap-2">
                <Sparkles className="w-5 h-5 text-accent mt-0.5 shrink-0" />
                <span>Wildlife conservation & Dian Fossey</span>
              </li>
              <li className="flex items-start gap-2">
                <Sparkles className="w-5 h-5 text-accent mt-0.5 shrink-0" />
                <span>Nuclear testing & environmental impacts</span>
              </li>
              <li className="flex items-start gap-2">
                <Sparkles className="w-5 h-5 text-accent mt-0.5 shrink-0" />
                <span>TensorFlow & optimization techniques</span>
              </li>
            </ul>
            <p className="text-lg font-medium pt-4">
              What would you like to search for today?
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap gap-4 pt-4">
            <Button
              onClick={() => onQuickStart("")}
              size="lg"
              className="flex-1 min-w-[200px] bg-primary hover:bg-primary/90"
            >
              <Search className="w-4 h-4 mr-2" />
              Start Searching
            </Button>
          </div>
        </Card>

        {/* Example queries */}
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground text-center">
            Try these example queries:
          </p>
          <div className="grid md:grid-cols-2 gap-3">
            {examples.map((example, index) => (
              <button
                key={index}
                onClick={() => onQuickStart(example)}
                className="text-left p-4 rounded-lg border border-border hover:border-accent hover:bg-accent/5 transition-colors text-sm"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
