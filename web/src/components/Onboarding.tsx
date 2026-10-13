import art from 'virtual:art'

export function Onboarding({ onDone }: { onDone: () => void }) {
  return (
    <div className="scrim" role="dialog" aria-modal="true" aria-label="Welcome to Curbcut">
      <div className="sheet">
        <img className="appicon" src={art['icons/app']?.url} width={56} height={56} alt="" />
        <h2>Curbcut</h2>
        <p>A shared map of curb cuts, rough pavement and blocked ramps, and routes that go around them.</p>
        <ul className="plain">
          <li><b>Record</b> a walk or roll and your phone tags rough ground and bumps.</li>
          <li><b>Photograph</b> a curb and your phone says whether the ramp is there, missing, damaged or blocked.</li>
          <li><b>Route</b> for a manual chair, a powered chair or a stroller.</li>
        </ul>
        <div className="card note">
          Sensors are read only while you are recording with the app open. Your trace stays on your phone. Faces are covered on the phone before a photo is sent. Licence plates are not found automatically, so there is a tool to cover them by hand.
        </div>
        <button className="primary" onClick={onDone} data-testid="onboarding-done">Get started</button>
      </div>
    </div>
  )
}
