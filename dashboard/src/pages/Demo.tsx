// The animated demo (story.html), framed under the top bar. It takes the keyboard on load,
// so ← → step through the scenes straight away.
const STORY = `${import.meta.env.BASE_URL}story.html`;

export function Demo() {
  return (
    <div className="demo">
      <iframe
        src={STORY}
        title="Itara, animated"
        allow="fullscreen"
        onLoad={(e) => e.currentTarget.contentWindow?.focus()}
      />
      <a className="demo-open" href={STORY} target="_blank" rel="noreferrer">Open full page</a>
    </div>
  );
}
