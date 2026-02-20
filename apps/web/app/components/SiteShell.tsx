import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="container">
      <nav className="nav">
        <Link href="/" className="badge">
          CardioRisk
        </Link>
        <div className="nav-links">
          <Link href="/assess">Assess</Link>
          <Link href="/results">Results</Link>
          <Link href="/about">Methodology</Link>
          <Link href="/privacy">Privacy</Link>
        </div>
        <Link href="/assess" className="button-primary">
          Start Assessment
        </Link>
      </nav>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="card" style={{ padding: "20px 24px", marginBottom: 24 }}>
          <strong>Clinical note:</strong> This tool supports decision-making and does not replace physician judgment.
        </div>
        <div>
          CardioRisk Platform. Evidence-based estimates with transparent inputs. No data stored by default.
        </div>
      </div>
    </footer>
  );
}
