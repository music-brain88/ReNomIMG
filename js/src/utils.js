export function round (v, round_off) {
  return Math.round(v * round_off) / round_off
}

export function round_percent (v) {
  return Math.round(v * 100)
}

export function min (x, y) {
  return x < y ? x : y
}

export function max (x, y) {
  return x > y ? x : y
}
