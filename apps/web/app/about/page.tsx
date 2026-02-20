import { Microscope, ShieldCheck, Stethoscope } from "lucide-react";
import { SiteFooter, SiteHeader } from "../components/SiteShell";

export default function AboutPage() {
  return (
    <main>
      <SiteHeader />
      <section className="section">
        <div className="container static-grid">
          <div className="card-panel">
            <div className="mini-badge">Methodology</div>
            <h1 className="type-h1">How the risk estimate is built</h1>
            <p className="lead">
              The platform uses established cardiovascular predictors inspired by Framingham and Pooled Cohort
              frameworks, then translates them into a transparent, interpretable score for decision support.
            </p>
          </div>
          <div className="card-panel stack">
            <div className="info-row">
              <Microscope size={18} />
              <div>
                <strong>Evidence-informed features</strong>
                <p>Age, sex, blood pressure, lipids, diabetes, smoking, and activity shape the estimate.</p>
              </div>
            </div>
            <div className="info-row">
              <Stethoscope size={18} />
              <div>
                <strong>Clinical context first</strong>
                <p>Score bands help prioritize next steps, but clinician judgment remains the final authority.</p>
              </div>
            </div>
            <div className="info-row">
              <ShieldCheck size={18} />
              <div>
                <strong>Transparent limitations</strong>
                <p>Inputs marked as unknown reduce precision and should be replaced with measured values.</p>
              </div>
            </div>
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}

