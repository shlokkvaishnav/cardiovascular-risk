import { DatabaseZap, EyeOff, LockKeyhole } from "lucide-react";
import { SiteFooter, SiteHeader } from "../components/SiteShell";

export default function PrivacyPage() {
  return (
    <main>
      <SiteHeader />
      <section className="section">
        <div className="container static-grid">
          <div className="card-panel">
            <div className="mini-badge">Privacy</div>
            <h1 className="type-h1">Zero-storage by design</h1>
            <p className="lead">
              Assessment answers remain in your browser session and are not persisted to platform servers.
            </p>
          </div>
          <div className="card-panel stack">
            <div className="info-row">
              <DatabaseZap size={18} />
              <div>
                <strong>No backend storage</strong>
                <p>Inputs are used only to compute the current on-screen result.</p>
              </div>
            </div>
            <div className="info-row">
              <EyeOff size={18} />
              <div>
                <strong>No hidden tracking of health entries</strong>
                <p>We do not profile users from medical form content.</p>
              </div>
            </div>
            <div className="info-row">
              <LockKeyhole size={18} />
              <div>
                <strong>Session-level visibility only</strong>
                <p>Closing the tab clears assessment context unless exported manually by the user.</p>
              </div>
            </div>
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}

