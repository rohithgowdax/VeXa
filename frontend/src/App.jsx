import { useEffect, useState } from "react";
import Translator from "./components/Translator";

const INTRO_HOLD_MS = 2600;
const INTRO_EXIT_MS = 760;
const INTRO_WORDS = ["Translate", "Reimagine", "Create"];

export default function App() {
  const [showIntro, setShowIntro] = useState(true);
  const [isExitingIntro, setIsExitingIntro] = useState(false);
  const [mountApp, setMountApp] = useState(false);
  const [wordIndex, setWordIndex] = useState(0);

  useEffect(() => {
    const exitTimer = setTimeout(() => {
      setIsExitingIntro(true);
      setMountApp(true);
    }, INTRO_HOLD_MS);

    const hideTimer = setTimeout(() => {
      setShowIntro(false);
    }, INTRO_HOLD_MS + INTRO_EXIT_MS + 60);

    return () => {
      clearTimeout(exitTimer);
      clearTimeout(hideTimer);
    };
  }, []);

  useEffect(() => {
    if (!showIntro || isExitingIntro) {
      return undefined;
    }

    const ticker = setInterval(() => {
      setWordIndex((prev) => (prev + 1) % INTRO_WORDS.length);
    }, 520);

    return () => {
      clearInterval(ticker);
    };
  }, [showIntro, isExitingIntro]);

  return (
    <>
      {showIntro && (
        <div className={`vexa-intro ${isExitingIntro ? "vexa-intro-exit" : ""}`}>
          <div className="vexa-intro-lights" aria-hidden="true">
            <span className="vexa-intro-orbit vexa-orbit-one" />
            <span className="vexa-intro-orbit vexa-orbit-two" />
            <span className="vexa-intro-flare" />
            <span className="vexa-intro-flare vexa-intro-flare-soft" />
            <div className="vexa-intro-stars">
              {Array.from({ length: 14 }).map((_, index) => (
                <span key={index} className="vexa-intro-star" style={{ "--i": index }} />
              ))}
            </div>
          </div>

          <svg viewBox="0 0 180 180" className="vexa-intro-sigil" aria-hidden="true">
            <defs>
              <linearGradient id="vexaSigilStroke" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#67e8f9" />
                <stop offset="100%" stopColor="#a78bfa" />
              </linearGradient>
            </defs>
            <circle cx="90" cy="90" r="68" className="vexa-sigil-ring" />
            <path
              d="M46 104L74 56L91 84L107 56L134 104"
              className="vexa-sigil-mark"
              fill="none"
              stroke="url(#vexaSigilStroke)"
              strokeWidth="9"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>

          <div className="vexa-intro-logo">Vexa</div>
          <p className="vexa-intro-tagline">Creative neural translation</p>
          <p className="vexa-intro-verb">{INTRO_WORDS[wordIndex]}</p>
        </div>
      )}

      {mountApp && (
        <div className={`vexa-app-shell ${isExitingIntro || !showIntro ? "vexa-app-visible" : ""}`}>
          <Translator />
        </div>
      )}
    </>
  );
}
