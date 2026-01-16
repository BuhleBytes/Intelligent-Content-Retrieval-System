import { Menu, X } from "lucide-react";
import { useEffect, useState } from "react";
import { ChatInterface } from "./components/chat-interface";
import { ConfigSidebar } from "./components/config-sidebar";
import { Button } from "./components/ui/button";
import { WelcomeScreen } from "./components/welcome-screen";

function App() {
  const [showWelcome, setShowWelcome] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [searchMode, setSearchMode] = useState("semantic");
  const [numResults, setNumResults] = useState(5);
  const [categories, setCategories] = useState(["all"]);
  const [keywords, setKeywords] = useState("");
  const [semanticWeight, setSemanticWeight] = useState(70);

  // Chat management state
  const [currentChatId, setCurrentChatId] = useState(null);
  const [currentMessages, setCurrentMessages] = useState([]);
  const [savedChats, setSavedChats] = useState([]);
  const [prefilledQuery, setPrefilledQuery] = useState("");

  // Load saved chats from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem("intelligent-search-chats");
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setSavedChats(parsed);
      } catch (e) {
        console.error("Failed to load chats:", e);
      }
    }
  }, []);

  useEffect(() => {
    if (savedChats.length > 0) {
      localStorage.setItem(
        "intelligent-search-chats",
        JSON.stringify(savedChats)
      );
    } else {
      // Clear localStorage if no chats
      localStorage.removeItem("intelligent-search-chats");
    }
  }, [savedChats]);

  const handleQuickStart = (exampleQuery) => {
    setShowWelcome(false);

    // Don't call handleNewChat here - let it happen naturally
    if (exampleQuery) {
      setPrefilledQuery(exampleQuery);
    }

    // Only create new chat if no current chat exists
    if (!currentChatId) {
      const newChatId = `Chat 1`;
      setCurrentChatId(newChatId);
      setCurrentMessages([]);
    }
  };

  const handleConfigure = () => {
    setShowWelcome(false);
    setSidebarOpen(true);

    // Only create new chat if no current chat exists
    if (!currentChatId) {
      const newChatId = `Chat 1`;
      setCurrentChatId(newChatId);
      setCurrentMessages([]);
    }
  };

  const handleNewChat = () => {
    console.log("🆕 Creating new chat...");
    console.log("Current chat ID:", currentChatId);
    console.log("Current messages:", currentMessages.length);
    console.log("Saved chats before:", savedChats.length);

    if (currentChatId && currentMessages.length > 0) {
      const chatToSave = {
        id: currentChatId,
        name: currentChatId,
        messages: currentMessages,
        timestamp: new Date().toISOString(),
      };

      setSavedChats((prev) => {
        // Check if this chat already exists
        const existingIndex = prev.findIndex(
          (chat) => chat.id === currentChatId
        );

        if (existingIndex !== -1) {
          // Update existing chat
          const updated = [...prev];
          updated[existingIndex] = chatToSave;
          console.log("✅ Updated existing chat:", currentChatId);
          return updated;
        } else {
          // Add new chat
          console.log("✅ Added new chat:", currentChatId);
          return [...prev, chatToSave];
        }
      });
    } else {
      console.log("⚠️ Current chat has no messages, not saving");
    }

    // Wait for state to update, then create new chat
    setTimeout(() => {
      setSavedChats((currentSaved) => {
        // Find highest chat number
        let maxChatNum = 0;
        currentSaved.forEach((chat) => {
          const match = chat.id.match(/Chat (\d+)/);
          if (match) {
            const num = parseInt(match[1]);
            if (num > maxChatNum) maxChatNum = num;
          }
        });

        const nextChatNumber = maxChatNum + 1;
        const newChatId = `Chat ${nextChatNumber}`;

        console.log("🎯 Creating new chat:", newChatId);
        setCurrentChatId(newChatId);
        setCurrentMessages([]);

        return currentSaved; // Don't modify saved chats here
      });
    }, 0);
  };

  const handleLoadChat = (chatId) => {
    console.log("📂 Loading chat:", chatId);

    // Save current chat before switching (only if it has messages)
    if (
      currentChatId &&
      currentMessages.length > 0 &&
      currentChatId !== chatId
    ) {
      const chatToSave = {
        id: currentChatId,
        name: currentChatId,
        messages: currentMessages,
        timestamp: new Date().toISOString(),
      };

      setSavedChats((prev) => {
        const existingIndex = prev.findIndex(
          (chat) => chat.id === currentChatId
        );
        if (existingIndex !== -1) {
          const updated = [...prev];
          updated[existingIndex] = chatToSave;
          return updated;
        }
        return [...prev, chatToSave];
      });
    }

    // Load selected chat
    const chatToLoad = savedChats.find((chat) => chat.id === chatId);
    if (chatToLoad) {
      console.log(
        "✅ Found chat, loading messages:",
        chatToLoad.messages.length
      );

      // Convert timestamp strings back to Date objects
      const messagesWithDates = chatToLoad.messages.map((msg) => ({
        ...msg,
        timestamp: new Date(msg.timestamp),
      }));

      setCurrentChatId(chatToLoad.id);
      setCurrentMessages(messagesWithDates);
    } else {
      console.error("❌ Chat not found:", chatId);
    }
  };

  const handleClearHistory = () => {
    console.log("🗑️ Clearing all history");
    setSavedChats([]);
    localStorage.removeItem("intelligent-search-chats");

    // Start completely fresh
    setCurrentChatId("Chat 1");
    setCurrentMessages([]);
  };

  if (showWelcome) {
    return (
      <WelcomeScreen
        onQuickStart={handleQuickStart}
        onConfigure={handleConfigure}
      />
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* TOGGLE BUTTON */}
      <Button
        variant="outline"
        size="icon"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className={`fixed top-4 z-50 bg-background shadow-lg transition-all duration-300 ${
          sidebarOpen ? "left-[304px]" : "left-4"
        }`}
      >
        {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-80" : "w-0"
        } relative border-r border-border bg-sidebar transition-all duration-300 ease-in-out overflow-hidden flex-shrink-0`}
      >
        <div className="w-80 h-full">
          <ConfigSidebar
            searchMode={searchMode}
            setSearchMode={setSearchMode}
            numResults={numResults}
            setNumResults={setNumResults}
            categories={categories}
            setCategories={setCategories}
            keywords={keywords}
            setKeywords={setKeywords}
            semanticWeight={semanticWeight}
            setSemanticWeight={setSemanticWeight}
            onNewChat={handleNewChat}
            savedChats={savedChats}
            currentChatId={currentChatId}
            onLoadChat={handleLoadChat}
            onClearHistory={handleClearHistory}
          />
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <ChatInterface
          searchMode={searchMode}
          numResults={numResults}
          categories={categories}
          keywords={keywords}
          semanticWeight={semanticWeight}
          messages={currentMessages}
          setMessages={setCurrentMessages}
          currentChatId={currentChatId}
          prefilledQuery={prefilledQuery}
          setPrefilledQuery={setPrefilledQuery}
        />
      </div>
    </div>
  );
}

export default App;
