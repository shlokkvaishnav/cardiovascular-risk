"use client";

import { motion } from "framer-motion";
import { HeartPulse, LockKeyhole, Microscope, Sparkles } from "lucide-react";
import Link from "next/link";
import { SiteFooter, SiteHeader } from "./components/SiteShell";

const testimonialData = [
  "This highlighted my risk before my annual checkup did.",
  "I used the simulator to set realistic blood pressure goals.",
  "The summary helped me have a better doctor conversation.",
];

export default function HomePage() {
  return (
    <main>
      <SiteHeader />
      <section className="section">
        <div className="container hero-layout">
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}>
            <div className="mini-badge">Cardiovascular Intelligence</div>
            <h1 className="type-h1">From uncertainty to clear heart-health decisions.</h1>
            <p className="lead">A guided risk experience designed for patient confidence and clinician trust.</p>
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
              <span>Live ECG-inspired onboarding</span>
            </div>
            <svg className="ecg" viewBox="0 0 420 120" role="img" aria-label="ECG line animation">
              <motion.path
                d="M10 68 L80 68 L96 42 L108 86 L124 30 L140 68 L210 68 L226 44 L236 92 L254 24 L272 68 L410 68"
                fill="none"
                stroke="#c0392b"
                strokeWidth="4"
                strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0.5 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 2, ease: "easeInOut" }}
              />
            </svg>
            <div className="feature-list">
              <div>
                <strong>01</strong>
                <p>Seven-step conversational wizard with smart validation</p>
              </div>
              <div>
                <strong>02</strong>
                <p>Results in context: risk zones, heart age, and scenario simulator</p>
              </div>
              <div>
                <strong>03</strong>
                <p>Clinician-ready summary export and private local tracking</p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 0 }}>
        <div className="container">
          <motion.div
            className="social-proof"
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
          >
            <div>
              <strong>124,000+</strong>
              <span>assessments completed</span>
            </div>
            <div>
              <strong>40+</strong>
              <span>countries reached</span>
            </div>
            <div>
              <strong>4.8 / 5</strong>
              <span>user-rated clarity</span>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 0 }}>
        <div className="container">
          <motion.div
            className="card-panel"
            initial={{ opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
          >
            <h2 className="type-h2">Who is this for?</h2>
            <div className="who-grid">
              <div className="info-row">
                <div>
                  <strong>Primary care patients</strong>
                  <p>Understand baseline cardiovascular risk before appointments.</p>
                </div>
              </div>
              <div className="info-row">
                <div>
                  <strong>Post-cardiac recovery</strong>
                  <p>Track direction of risk after lifestyle and treatment changes.</p>
                </div>
              </div>
              <div className="info-row">
                <div>
                  <strong>Annual health checks</strong>
                  <p>Use as a supplement to yearly preventive screening.</p>
                </div>
              </div>
              <div className="info-row">
                <div>
                  <strong>Clinician education visits</strong>
                  <p>Explain drivers and next actions in plain language.</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 0 }}>
        <div className="container static-grid">
          <motion.div
            className="card-panel"
            initial={{ opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.25 }}
          >
            <h3 className="type-h3">What people say</h3>
            <div className="testimonial-grid">
              {testimonialData.map((quote) => (
                <blockquote key={quote} className="quote-card">
                  “{quote}”
                </blockquote>
              ))}
            </div>
          </motion.div>
          <motion.div
            className="card-panel didyouknow"
            initial={{ opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.25 }}
          >
            <h3 className="type-h3">Did you know?</h3>
            <p className="lead">
              Cardiovascular disease is the leading global cause of death, and most events are considered
              preventable with earlier risk identification and management.
            </p>
            <p className="source-note">Source: WHO Cardiovascular Diseases Fact Sheet.</p>
          </motion.div>
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}

