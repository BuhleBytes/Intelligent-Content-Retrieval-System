import { Bot, Settings as SettingsIcon, User } from "lucide-react";
import { Card } from "./ui/card";

export function ChatMessage({ message }) {
  const formatTime = (date) => {
    // ✅ FIX: Handle both Date objects and strings
    const dateObj = date instanceof Date ? date : new Date(date);

    // Check if valid date
    if (isNaN(dateObj.getTime())) {
      return "Invalid time";
    }

    return dateObj.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (message.type === "user") {
    return (
      <Card className="p-4 ml-auto max-w-[85%] bg-primary/5 border-primary/20">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center shrink-0">
            <User className="w-4 h-4 text-primary-foreground" />
          </div>
          <div className="flex-1 space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium text-sm">You</span>
              <span className="text-xs text-muted-foreground">
                {formatTime(message.timestamp)}
              </span>
            </div>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content}
            </p>
          </div>
        </div>
      </Card>
    );
  }

  if (message.type === "system") {
    return (
      <Card className="p-4 max-w-[85%] bg-muted/50 border-muted">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-full bg-muted-foreground/20 flex items-center justify-center shrink-0">
            <SettingsIcon className="w-4 h-4 text-muted-foreground" />
          </div>
          <div className="flex-1 space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium text-sm">System</span>
              <span className="text-xs text-muted-foreground">
                {formatTime(message.timestamp)}
              </span>
            </div>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content}
            </p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-4 max-w-[85%] bg-accent/5 border-accent/20">
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center shrink-0">
          <Bot className="w-4 h-4 text-accent-foreground" />
        </div>
        <div className="flex-1 space-y-1">
          <div className="flex items-center justify-between gap-2">
            <span className="font-medium text-sm">AI Assistant</span>
            <span className="text-xs text-muted-foreground">
              {formatTime(message.timestamp)}
            </span>
          </div>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>
      </div>
    </Card>
  );
}
