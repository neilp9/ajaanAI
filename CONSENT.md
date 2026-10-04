# Consent record — voice and persona

This project builds an AI that speaks in the manner of Thanissaro Bhikkhu (Ajaan Geoff), trained on
his public talks. Using a **clone of his voice** requires his permission (or that of Metta Forest
Monastery acting for him). ElevenLabs also requires that a Professional Voice Clone be created and
verified **on the speaker's own account**, then shared with ours.

`ajaanai agent-sync` refuses to use a cloned voice (`AGENT_VOICE_IS_CLONE=true`) until the fields
below show granted permission. Until then the agent uses a stock voice and its greeting says it
is an AI, not Ajaan Geoff.

```
status: pending
# pending | granted | declined
documentation: none
# where the written permission is kept (e.g. email date + sender, signed letter location)
granted_by:
scope:
# e.g. "voice clone for a free phone line; persona trained on dhammatalks.org content"
conditions:
# anything requested: disclaimers, review, takedown process, no commercial use, ...
date:
```

Independent of the voice, the site's content is licensed CC BY-NC 4.0, and the author treats
*any* sale as commercial use. Keep the service free, and keep the attribution in the greeting.
