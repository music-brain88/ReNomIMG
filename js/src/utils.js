export function round(v, round_off) {
  return Math.round(v*round_off) / round_off;
}

export function round_percent(v) {
  return Math.round(v*100);
}