"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { UserPlus } from "lucide-react";
import { SiteFooter, SiteHeader } from "../components/SiteShell";
import { register, AuthApiError } from "../lib/authApi";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await register(email, password);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof AuthApiError ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <SiteHeader />
      <section className="section">
        <div className="container" style={{ maxWidth: 420 }}>
          <motion.div className="card-panel stack" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <div className="mini-badge">Account</div>
            <h1 className="type-h2">Create an account</h1>
            <p className="lead">
              Optional -- lets you save assessments and track your risk trend across visits. Nothing here affects
              the zero-storage guest flow.
            </p>
            <form onSubmit={onSubmit} className="stack">
              <div className="field">
                <label htmlFor="email" className="field-label">Email</label>
                <div className="floating">
                  <input
                    id="email"
                    type="email"
                    required
                    className="floating-input"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </div>
              </div>
              <div className="field">
                <label htmlFor="password" className="field-label">Password</label>
                <div className="floating">
                  <input
                    id="password"
                    type="password"
                    required
                    minLength={8}
                    className="floating-input"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                </div>
                <p className="helper-inline">At least 8 characters.</p>
              </div>
              {error && (
                <div className="error-box" aria-live="assertive">
                  <p>{error}</p>
                </div>
              )}
              <button type="submit" className="btn btn-primary" disabled={loading}>
                <UserPlus size={16} /> {loading ? "Creating account..." : "Create account"}
              </button>
            </form>
            <p className="lead">
              Already have an account? <Link href="/login">Log in</Link>
            </p>
          </motion.div>
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}
