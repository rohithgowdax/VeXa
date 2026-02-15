import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

export async function translateText(text) {
  const { data } = await api.post("/translate", { text });
  return data.translation;
}
