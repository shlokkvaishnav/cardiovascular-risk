"use client";

import { motion } from "framer-motion";
import { HeartPulse, LockKeyhole, Microscope, Sparkles } from "lucide-react";
import Link from "next/link";
import { SiteFooter, SiteHeader } from "./components/SiteShell";

export default function HomePage() {
  return (
    <main>
      <SiteHeader />
      <section className="section">
        <div className="container hero-layout">
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}>
            <div className="mini-badge">Cardiovascular Intelligence</div>
            <h1 className="type-h1">From uncertainty to clear heart-health decisions.</h1>
            <p className="lead">
              A modern, guided risk assessment designed for patient confidence and clinician trust.
            </p>
            <div className="cta-row">
              <Link href="/assess" className="btn btn-primary">
                Start Assessment
              </Link>
              <Link href="/about" className="btn btn-subtle">
                Explore Methodology
              </Link>
            </div>
            <div className="trust-strip">
              <span>
                <LockKeyhole size={15} />
                No data stored
              </span>
              <span>
                <Microscope size={15} />
                Evidence-aligned
              </span>
              <span>
                <Sparkles size={15} />
                Personalized actions
              </span>
            </div>
          </motion.div>
          <motion.div
            className="card-panel hero-panel"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <div className="hero-panel-head">
              <HeartPulse size={18} />
              <span>7-step Conversational Wizard</span>
            </div>
            <div className="feature-list">
              <div>
                <strong>01</strong>
                <p>Welcome with privacy assurance and trust cues</p>
              </div>
              <div>
                <strong>02</strong>
                <p>Smart inputs with units, floating labels, and inline validation</p>
              </div>
              <div>
                <strong>03</strong>
                <p>Interactive dashboard with risk zones and action guidance</p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}

