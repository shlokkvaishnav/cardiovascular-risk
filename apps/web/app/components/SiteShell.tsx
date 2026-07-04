"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ThemeToggle } from "./ThemeToggle";
import { isLoggedIn } from "../lib/authApi";

export function SiteHeader() {
  // Checked post-mount only (localStorage isn't available during SSR, and
  // checking it during render would cause a hydration mismatch).
  const [loggedIn, setLoggedIn] = useState(false);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reads localStorage, only available post-mount
    setLoggedIn(isLoggedIn());
  }, []);

  return (
    <header className="header-wrap">
      <nav className="container nav">
        <Link href="/" className="brand" aria-label="CardioRisk Home">
          <span className="brand-mark">CR</span>
          <span>CardioRisk</span>
        </Link>
        <div className="nav-links" aria-label="Primary Navigation">
          <Link href="/assess">Assess</Link>
          <Link href="/results">Results</Link>
          <Link href="/about">Methodology</Link>
          <Link href="/privacy">Privacy</Link>
          <Link href={loggedIn ? "/dashboard" : "/login"}>{loggedIn ? "Dashboard" : "Log in"}</Link>
        </div>
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          <ThemeToggle />
          <Link href="/assess" className="btn btn-primary nav-cta">
            Start Assessment
          </Link>
        </div>
      </nav>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="footer section">
      <div className="container">
        <div className="card-panel" style={{ marginBottom: 20 }}>
          <strong>Clinical note:</strong> This tool supports decision-making and does not replace physician judgment.
        </div>
        <div className="footer-copy">
          CardioRisk Platform. Evidence-based estimates with transparent inputs. No data stored by default.
        </div>
      </div>
    </footer>
  );
}

