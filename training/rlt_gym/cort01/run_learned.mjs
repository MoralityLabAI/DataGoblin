import { campaign2, tests } from "./learned.mjs";
console.error(tests());
console.log(JSON.stringify([27012026,27012027,27012028].map(campaign2), null, 2));
