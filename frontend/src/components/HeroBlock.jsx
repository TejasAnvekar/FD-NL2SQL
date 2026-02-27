//frontend/src/components/HeroBlock.jsx
export default function HeroBlock() {
  return (
    <section className="hero-block">
      <div className="hero-block-inner">
        <div className="hero-block-badge">
          {/* TODO: Replace badge label */}
          Living Systematic Review
        </div>
        {/* TODO: Replace with your actual heading */}
        <h2 className="hero-block-heading">Test Heading: About This Database</h2>
        <div className="hero-block-body">
          <p>
            {/* TODO: Replace this paragraph with your actual introduction/description */}
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
            labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
            laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
            voluptate velit esse cillum dolore eu fugiat nulla pariatur.
          </p>
          <p>
            {/* TODO: Replace this paragraph with methods/scope description */}
            Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim
            id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
            doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis.
          </p>
        </div>
        <div className="hero-block-pills">
          {/* TODO: Replace with your actual keywords/scope tags */}
          <span className="hero-pill">Phase 2 &amp; 3 RCTs</span>
          <span className="hero-pill">PD-1 / PD-L1 / CTLA-4</span>
          <span className="hero-pill">19 Cancer Types</span>
          <span className="hero-pill">Peer-reviewed Sources</span>
        </div>
      </div>
    </section>
  )
}