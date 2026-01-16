const API_BASE_URL = "https://web-production-f2b40.up.railway.app";

class APIService {
  /**
   * Get the API base URL
   */
  get baseURL() {
    return API_BASE_URL;
  }

  /**
   * Health check - verify backend is running
   */
  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) throw new Error("Health check failed");
      return await response.json();
    } catch (error) {
      console.error("Health check error:", error);
      throw error;
    }
  }

  /**
   * Get database statistics
   */
  async getStats() {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      if (!response.ok) throw new Error("Failed to fetch stats");
      return await response.json();
    } catch (error) {
      console.error("Stats error:", error);
      throw error;
    }
  }

  /**
   * Perform SEMANTIC search
   * @param {Object} params
   * @param {string} params.query - Search query text
   * @param {number} params.n_results - Number of results (1-20)
   * @param {string} params.filter_category - Category filter (optional)
   * @param {boolean} params.enable_llm - Enable AI enhancement (optional)
   */
  async semanticSearch(params) {
    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: params.query,
          n_results: params.n_results,
          filter_category: params.filter_category,
          enable_llm: params.enable_llm || false, // NEW
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Semantic search failed");
      }
      return await response.json();
    } catch (error) {
      console.error("Semantic search error:", error);
      throw error;
    }
  }

  /**
   * Perform HYBRID search
   * @param {Object} params
   * @param {string} params.query - Search query text
   * @param {Array} params.keywords - Keywords for matching
   * @param {number} params.n_results - Number of results (1-20)
   * @param {string} params.filter_category - Category filter (optional)
   * @param {number} params.semantic_weight - Semantic weight (0.0-1.0)
   * @param {number} params.keyword_weight - Keyword weight (0.0-1.0)
   * @param {boolean} params.enable_llm - Enable AI enhancement (optional)
   */
  async hybridSearch(params) {
    try {
      const response = await fetch(`${API_BASE_URL}/hybrid`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: params.query,
          keywords: params.keywords,
          n_results: params.n_results,
          filter_category: params.filter_category,
          semantic_weight: params.semantic_weight,
          keyword_weight: params.keyword_weight,
          enable_llm: params.enable_llm || false, // NEW
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Hybrid search failed");
      }
      return await response.json();
    } catch (error) {
      console.error("Hybrid search error:", error);
      throw error;
    }
  }

  /**
   * Enhance specific chunks on-demand
   * @param {Object} params
   * @param {string} params.query - Original search query
   * @param {Array} params.results - Full results array
   * @param {Array} params.indices - Indices of chunks to enhance
   */
  async enhanceRemaining(params) {
    try {
      const response = await fetch(`${API_BASE_URL}/enhance`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: params.query,
          results: params.results,
          indices: params.indices,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Enhancement failed");
      }
      return await response.json();
    } catch (error) {
      console.error("Enhancement error:", error);
      throw error;
    }
  }
}

// Export a singleton instance
export const apiService = new APIService();
export default apiService;
