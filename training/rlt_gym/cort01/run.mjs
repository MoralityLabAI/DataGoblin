import { campaign, selfTest } from "./gym.mjs";
console.error(selfTest());
console.log(JSON.stringify(campaign(), null, 2));
