"""Word lists behind the lexical and pragmatic features. Sources are noted per list so a reader can
check what each feature measures. Lists are lowercase; multi-word entries are matched as phrases."""

# Function words: the classic stylometric fingerprint (Mosteller & Wallace; Burrows' Delta uses the
# most frequent words, which are overwhelmingly these). Content-free by construction.
FUNCTION_WORDS = sorted(set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also although however therefore though thus yet whether upon within
without across among along around behind beside beyond toward towards via
""".split()))

# Hedges and boosters: Hyland (2005), Metadiscourse; trimmed to forms common in conversation.
HEDGES = [
    "maybe", "perhaps", "possibly", "probably", "might", "may", "could", "seems", "seem", "appears", "apparently",
    "i think", "i guess", "i suppose", "i believe", "i feel", "sort of", "kind of", "a bit", "a little", "somewhat",
    "fairly", "rather", "quite", "likely", "unlikely", "roughly", "about", "around", "tend to", "in general",
    "not sure", "if possible", "hopefully", "presumably", "arguably",
]
BOOSTERS = [
    "definitely", "certainly", "clearly", "obviously", "absolutely", "of course", "surely", "undoubtedly", "always",
    "never", "really", "truly", "totally", "completely", "exactly", "for sure", "no doubt", "indeed", "in fact",
    "actually", "literally", "honestly",
]
INTENSIFIERS = ["so", "very", "really", "super", "extremely", "incredibly", "totally", "absolutely", "hugely", "massively", "such", "too", "ever so", "properly"]

# Discourse markers (Schiffrin 1987; Fraser 1999).
DISCOURSE_MARKERS = [
    "so", "well", "anyway", "anyways", "oh", "ah", "okay", "ok", "right", "now", "then", "also", "plus", "but",
    "and", "because", "actually", "basically", "honestly", "by the way", "in fact", "you know", "i mean", "like",
    "just so you know", "fyi", "to be honest", "tbh", "that said", "having said that",
]

# British vs American spelling pairs (a register signal for UK brands such as Kiki).
UK_US = [
    ("colour", "color"), ("favourite", "favorite"), ("organise", "organize"), ("organised", "organized"), ("realise", "realize"),
    ("apologise", "apologize"), ("recognise", "recognize"), ("centre", "center"), ("theatre", "theater"), ("travelling", "traveling"),
    ("travelled", "traveled"), ("cancelled", "canceled"), ("cancelling", "canceling"), ("behaviour", "behavior"), ("honour", "honor"),
    ("neighbour", "neighbor"), ("programme", "program"), ("cheque", "check"), ("licence", "license"), ("defence", "defense"),
    ("jewellery", "jewelry"), ("grey", "gray"), ("mum", "mom"), ("flat", "apartment"), ("holiday", "vacation"),
    ("whilst", "while"), ("amongst", "among"), ("learnt", "learned"), ("fulfil", "fulfill"), ("enrol", "enroll"),
]

CONTRACTIONS_RE = r"\b\w+(?:'|’)(?:s|re|ve|ll|d|m|t)\b|\b(?:can't|won't|don't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|doesn't|didn't|couldn't|shouldn't|wouldn't|mustn't)\b"
EXPANDABLE = ["do not", "does not", "did not", "is not", "are not", "was not", "were not", "have not", "has not", "cannot", "can not", "will not", "would not", "should not", "could not", "i am", "you are", "we are", "they are", "it is", "that is", "i will", "we will", "you will", "i have", "we have", "i would", "we would"]

# Informal register markers (textese, slang, filled pauses).
TEXTESE = ["lol", "haha", "hahaha", "omg", "tbh", "imo", "imho", "btw", "idk", "ikr", "ngl", "fyi", "pls", "plz", "thx", "ty", "u", "ur", "r", "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep", "yup", "nope", "nah", "ok", "okay", "cool", "awesome", "amazing", "lovely", "cheers", "mate", "x", "xx", "xxx"]

PROFANITY = ["damn", "hell", "shit", "fuck", "fucking", "bloody", "crap", "bastard", "arse", "ass", "piss"]

# Social acts: high-precision cue phrases. The NLI model decides; cues give a fast, auditable prior
# and a tie-breaker (see features/pragmatics.py).
SOCIAL_ACT_CUES = {
    "greet": [r"^(hi|hey|hello|hiya|good (morning|afternoon|evening)|morning|evening)\b", r"^dear\b"],
    "thank": [r"\bthank(s| you)\b", r"\bcheers\b", r"\bappreciate(d)?\b", r"\bta\b"],
    "apologise": [r"\b(sorry|apologi[sz]e|apologies|my bad|excuse me)\b"],
    "offer": [r"\b(happy to|glad to|we can|i can|let me|would you like|shall i|we could|i could|feel free)\b"],
    "request": [r"\b(could you|can you|would you|please|pls|plz|do you mind|let us know|let me know|send us|dm us)\b"],
    "reassure": [r"\b(don'?t worry|no worries|no problem|not a problem|you'?re (all )?set|we'?ve got (you|this)|it'?s (all )?fine|rest assured|all good)\b"],
    "refuse": [r"\b(unfortunately|i'?m afraid|we can'?t|we cannot|not able to|unable to|won'?t be able|isn'?t possible|no longer)\b"],
    "close": [r"\b(have a (great|good|lovely|nice) (day|one|weekend|evening)|take care|speak soon|talk soon|best wishes|kind regards|best,|thanks again)\b"],
    "empathise": [r"\b(i understand|we understand|that sounds|i can imagine|must be (frustrating|annoying|hard)|totally get|completely understand|i hear you)\b"],
    "praise": [r"\b(great|amazing|awesome|love|lovely|brilliant|fantastic|wonderful) (question|idea|choice|news|to hear)\b"],
}

# Social acts scored by the NLI model: hypothesis per act (zero-shot, entailment probability).
SOCIAL_ACT_HYPOTHESES = {
    "greet": "The writer is greeting someone.",
    "thank": "The writer is thanking someone.",
    "apologise": "The writer is apologising.",
    "offer": "The writer is offering to help or to do something.",
    "request": "The writer is asking the reader to do something.",
    "reassure": "The writer is reassuring the reader that things are fine.",
    "refuse": "The writer is declining or saying something cannot be done.",
    "empathise": "The writer is showing empathy for how the reader feels.",
    "inform": "The writer is giving factual information.",
    "close": "The writer is ending the conversation politely.",
}
