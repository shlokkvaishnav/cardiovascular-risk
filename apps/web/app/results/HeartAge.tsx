"use client";

import { useEffect, useMemo, useState } from "react";

export function HeartAge(props: { heartAge: number; age: number }) {
  const { heartAge, age } = props;
  const [displayAge, setDisplayAge] = useState(0);

  useEffect(() => {
    let frame = 0;
    const totalFrames = 32;
    const timer = window.setInterval(() => {
      frame += 1;
      const next = Math.round((heartAge * frame) / totalFrames);
      setDisplayAge(next);
      if (frame >= totalFrames) {
        window.clearInterval(timer);
      }
    }, 28);

    return () => window.clearInterval(timer);
  }, [heartAge]);

  const delta = heartAge - age;
  const toneClass = delta <= 0 ? "heart-age-good" : "heart-age-bad";
  const deltaText = useMemo(() => {
    if (delta === 0) return "Matches your current age";
    if (delta > 0) return `+${delta} years older than you are`;
    return `${Math.abs(delta)} years younger than your age`;
  }, [delta]);

  return (
    <div className={`heart-age-card ${toneClass}`}>
      <p className="heart-age-title">Heart Age</p>
      <div className="heart-age-number mono">{displayAge}</div>
      <p className="heart-age-subtitle">Compared to your age: {age}</p>
      <span className="chip heart-age-chip">{deltaText}</span>
    </div>
  );
}

