import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import zlib from "node:zlib";

function parseArgs(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value == null) throw new Error("expected --name value arguments");
    values[key.slice(2)] = value;
  }
  return values;
}

function readGzipRows(filePath) {
  return zlib.gunzipSync(fs.readFileSync(filePath))
    .toString("utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map(JSON.parse);
}

function normalize(value) {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(normalize);
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, normalize(value[key])]));
}

function equal(left, right) {
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

const args = parseArgs(process.argv.slice(2));
for (const name of ["engine", "canonical", "decisions", "episode-index"]) {
  if (!args[name]) throw new Error(`missing --${name}`);
}

const enginePath = path.resolve(args.engine);
const canonicalPath = path.resolve(args.canonical);
const decisionPath = path.resolve(args.decisions);
const episodeIndex = Number(args["episode-index"]);
if (!Number.isSafeInteger(episodeIndex) || episodeIndex < 0) throw new Error("invalid --episode-index");

const { COURT_VERSION, createCourtModelEpisode } = await import(pathToFileURL(enginePath));
const canonicalRows = readGzipRows(canonicalPath);
const decisions = readGzipRows(decisionPath);
if (canonicalRows.length !== 1) throw new Error("Court canonical shard must contain exactly one episode");
const original = canonicalRows[0];
if (String(original.env_version) !== String(COURT_VERSION)) {
  process.stdout.write(`${JSON.stringify({
    schema_version: "court-environment-replay.v1",
    status: "INCONCLUSIVE",
    reason: "ENGINE_VERSION_MISMATCH",
    declared_engine_version: String(original.env_version),
    available_engine_version: String(COURT_VERSION),
  })}\n`);
  process.exit(0);
}

const decisionsByKey = new Map();
for (const decision of decisions) {
  const key = [decision.turn, decision.phase, decision.seat].join("|");
  if (decisionsByKey.has(key)) throw new Error("duplicate Court decision coordinate");
  decisionsByKey.set(key, decision);
}
const assignments = original.players.map((player) => ({
  seat: player.seat,
  disposition: player.disposition,
  disposition_prompt_id: player.disposition_prompt_id,
  model: player.generator_model,
  reasoning_effort: player.reasoning_effort,
  id: player.assignment_cell,
}));
const used = new Set();
const replay = await createCourtModelEpisode({
  seed: `mixed-court-${episodeIndex}`,
  episodeIndex,
  players: assignments.length,
  assignments,
  maxTurns: 16,
  generatorModel: "mixed Sol/Terra/Luna; per-player metadata",
  decide: async ({ player, turn, phase }) => {
    const key = [turn, phase, player.seat].join("|");
    const decision = decisionsByKey.get(key);
    if (!decision) throw new Error(`missing Court decision coordinate ${key}`);
    if (used.has(key)) throw new Error(`Court decision reused at ${key}`);
    used.add(key);
    return decision.target;
  },
});
const checks = {
  all_decisions_consumed: used.size === decisions.length,
  players_equal: equal(replay.episode.players, original.players),
  turns_equal: equal(replay.episode.turns, original.turns),
  outcome_equal: equal(replay.episode.outcome, original.outcome),
};
process.stdout.write(`${JSON.stringify({
  schema_version: "court-environment-replay.v1",
  status: Object.values(checks).every(Boolean) ? "PASS" : "INCONCLUSIVE",
  reason: Object.values(checks).every(Boolean) ? "REPLAY_MATCH" : "REPLAY_MISMATCH",
  declared_engine_version: String(original.env_version),
  available_engine_version: String(COURT_VERSION),
  decisions: decisions.length,
  decisions_consumed: used.size,
  checks,
  canonical_sha256: sha256File(canonicalPath),
  decisions_sha256: sha256File(decisionPath),
  engine_sha256: sha256File(enginePath),
})}\n`);
