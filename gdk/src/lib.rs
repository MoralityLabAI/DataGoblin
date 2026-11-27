//! Data Goblin, a Goose GDK app for the DataGoblin research suite.
//!
//! A goose-agent state machine drives a local model through DataGoblin's hill climb: adapters on the
//! driver model itself, looped and recurrent modules, evals and rollouts built by bigger local models
//! (Agonist), and curation. The hill-climb MCP server (`python -m datagoblin.hillclimb.server`) owns
//! recipes, metrics, gates and the ledger; GDK skills (`gdk/skills/*/SKILL.md`) structure the climb.

pub mod agent;
pub mod mcp;
pub mod skills;
pub mod text_tools;
