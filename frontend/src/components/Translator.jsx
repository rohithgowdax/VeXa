import { useEffect, useRef, useState } from "react";
import { translateText } from "../api";

export default function Translator() {
  const [text, setText] = useState("");
  const [translation, setTranslation] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleTranslate = async () => {
    const cleaned = text.trim();
    if (!cleaned) {
      setError("Please enter German text to translate.");
      setTranslation("");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const result = await translateText(cleaned);
      setTranslation(result || "");
    } catch (err) {
      const message =
        err?.response?.data?.detail ||
        err?.message ||
        "Translation failed. Please try again.";
      setError(message);
      setTranslation("");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.page} className="vexa-page">
      <div style={styles.backgroundLayer} aria-hidden="true">
        <div style={{ ...styles.gradientBlob, ...styles.blobA }} className="vexa-blob" />
        <div style={{ ...styles.gradientBlob, ...styles.blobB }} className="vexa-blob" />
        <div style={{ ...styles.gradientBlob, ...styles.blobC }} className="vexa-blob" />
        <div style={styles.meshOverlay} className="vexa-mesh" />
      </div>

      <div style={styles.card} className="vexa-card">
        <div style={styles.heroRow} className="vexa-hero">
          <div>
            <p style={styles.kicker}>Creative neural translation</p>
            <h1 style={styles.title}>Vexa</h1>
            <p style={styles.subtitle}>
              Turn German sentences into fluent English with a clean artistic interface.
            </p>
          </div>

          <svg viewBox="0 0 180 120" style={styles.headerArt} className="vexa-header-art" aria-hidden="true">
            <defs>
              <linearGradient id="waveA" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#22d3ee" />
                <stop offset="100%" stopColor="#6366f1" />
              </linearGradient>
              <linearGradient id="waveB" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#f472b6" />
                <stop offset="100%" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
            <path
              d="M10 64C34 24 66 104 90 64C114 24 146 104 170 64"
              fill="none"
              stroke="url(#waveA)"
              strokeWidth="8"
              strokeLinecap="round"
            />
            <path
              d="M10 78C34 38 66 118 90 78C114 38 146 118 170 78"
              fill="none"
              stroke="url(#waveB)"
              strokeWidth="6"
              strokeLinecap="round"
              opacity="0.9"
            />
            <circle cx="32" cy="24" r="6" fill="#22d3ee" opacity="0.9" />
            <circle cx="148" cy="28" r="6" fill="#a78bfa" opacity="0.9" />
          </svg>
        </div>

        <div style={styles.workspace}>
          <div style={{ ...styles.panel, ...styles.panelLeft }} className="vexa-panel">
            <label htmlFor="source" style={styles.label}>
              German input
            </label>
            <textarea
              id="source"
              ref={inputRef}
              rows={9}
              style={styles.textarea}
              className="vexa-textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type German text here..."
            />
            <button
              style={{ ...styles.button, ...(isLoading ? styles.buttonLoading : {}) }}
              className="vexa-button"
              onClick={handleTranslate}
              disabled={isLoading}
            >
              {isLoading ? "Translating..." : "Translate"}
            </button>
            {isLoading && (
              <p style={styles.loading} className="vexa-loading">
                ⏳ Running model inference...
              </p>
            )}
            {error && <p style={styles.error}>{error}</p>}
          </div>

          <div style={{ ...styles.panel, ...styles.panelRight }} className="vexa-panel">
            <h2 style={styles.outputTitle}>English translation</h2>
            <div style={styles.outputBox} className="vexa-output">
              {translation || "Your translation will appear here."}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100dvh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "20px",
    boxSizing: "border-box",
    position: "relative",
    overflow: "hidden",
    background: "linear-gradient(160deg, #020617 0%, #0f172a 38%, #111827 100%)",
    backgroundSize: "160% 160%",
    animation: "vexaBackgroundShift 20s ease-in-out infinite alternate",
    fontFamily:
      'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  },
  backgroundLayer: {
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
  },
  gradientBlob: {
    position: "absolute",
    borderRadius: "9999px",
    filter: "blur(58px)",
    opacity: 0.44,
    willChange: "transform",
  },
  blobA: {
    width: "360px",
    height: "360px",
    top: "-110px",
    left: "-80px",
    background: "#0891b2",
    animation: "vexaBlobFloatA 14s ease-in-out infinite",
  },
  blobB: {
    width: "440px",
    height: "440px",
    bottom: "-190px",
    right: "-100px",
    background: "#7c3aed",
    animation: "vexaBlobFloatB 16s ease-in-out infinite",
  },
  blobC: {
    width: "280px",
    height: "280px",
    top: "36%",
    right: "28%",
    background: "#db2777",
    animation: "vexaBlobFloatC 13s ease-in-out infinite",
  },
  meshOverlay: {
    position: "absolute",
    inset: 0,
    backgroundImage:
      "linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px)",
    backgroundSize: "34px 34px",
    maskImage: "radial-gradient(circle at center, black 40%, transparent 95%)",
    animation: "vexaMeshDrift 28s linear infinite",
  },
  card: {
    width: "100%",
    maxWidth: "980px",
    position: "relative",
    zIndex: 1,
    background: "linear-gradient(140deg, rgba(255,255,255,0.95), rgba(238,242,255,0.9))",
    borderRadius: "28px",
    border: "1px solid rgba(226, 232, 240, 0.9)",
    padding: "32px",
    boxShadow: "0 30px 75px rgba(2, 6, 23, 0.5)",
    backdropFilter: "blur(12px)",
    animation: "vexaCardEnter 700ms cubic-bezier(0.2, 0.7, 0.2, 1)",
  },
  heroRow: {
    display: "flex",
    justifyContent: "space-between",
    gap: "24px",
    alignItems: "flex-start",
    marginBottom: "22px",
  },
  kicker: {
    margin: 0,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    fontSize: "0.7rem",
    color: "#4338ca",
    fontWeight: 700,
  },
  subtitle: {
    margin: "12px 0 0",
    color: "#334155",
    fontSize: "0.98rem",
    maxWidth: "56ch",
  },
  headerArt: {
    width: "180px",
    height: "120px",
    flexShrink: 0,
    animation: "vexaArtFloat 8s ease-in-out infinite",
  },
  title: {
    margin: 0,
    marginTop: "5px",
    fontSize: "2.1rem",
    color: "#0b1220",
    lineHeight: 1.08,
  },
  workspace: {
    display: "flex",
    gap: "18px",
    flexWrap: "wrap",
  },
  panel: {
    flex: "1 1 420px",
    background: "rgba(255,255,255,0.78)",
    border: "1px solid #e2e8f0",
    borderRadius: "18px",
    padding: "16px",
    boxShadow: "inset 0 1px 0 rgba(255,255,255,0.8)",
    transition: "transform 250ms ease, box-shadow 250ms ease",
    animation: "vexaPanelReveal 650ms ease both",
  },
  panelLeft: {
    animationDelay: "80ms",
  },
  panelRight: {
    animationDelay: "170ms",
  },
  label: {
    display: "block",
    fontWeight: 600,
    marginBottom: "8px",
    color: "#0f172a",
  },
  textarea: {
    width: "100%",
    boxSizing: "border-box",
    border: "1px solid #cbd5e1",
    borderRadius: "14px",
    background: "#f8fafc",
    padding: "14px",
    fontSize: "1rem",
    resize: "vertical",
    outline: "none",
    marginBottom: "12px",
    boxShadow: "inset 0 1px 2px rgba(15,23,42,0.05)",
    transition: "border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease",
  },
  button: {
    width: "100%",
    border: "none",
    borderRadius: "14px",
    padding: "13px 16px",
    background: "linear-gradient(92deg, #2563eb, #7c3aed)",
    color: "white",
    fontWeight: 700,
    cursor: "pointer",
    boxShadow: "0 12px 24px rgba(59, 130, 246, 0.32)",
    transition: "transform 180ms ease, box-shadow 180ms ease, filter 180ms ease",
  },
  buttonLoading: {
    animation: "vexaButtonPulse 1.4s ease-in-out infinite",
  },
  loading: {
    margin: "10px 0 0",
    color: "#334155",
    animation: "vexaFadePulse 1.6s ease-in-out infinite",
  },
  error: {
    margin: "10px 0 0",
    color: "#dc2626",
    fontWeight: 600,
  },
  outputTitle: {
    fontSize: "1.1rem",
    margin: "0 0 10px",
    color: "#0f172a",
  },
  outputBox: {
    minHeight: "214px",
    border: "1px solid #dbeafe",
    borderRadius: "14px",
    background: "linear-gradient(160deg, #f8fafc, #eef2ff)",
    padding: "14px",
    color: "#0f172a",
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
    transition: "transform 260ms ease, box-shadow 260ms ease",
  },
};
