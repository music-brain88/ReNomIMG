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

export function getTagColor (n) {
  if (n % 10 === 0) return '#E7009A'
  if (n % 10 === 1) return '#9F13C1'
  if (n % 10 === 2) return '#582396'
  if (n % 10 === 3) return '#0B20C4'
  if (n % 10 === 4) return '#3F9AAF'
  if (n % 10 === 5) return '#14884B'
  if (n % 10 === 6) return '#BBAA19'
  if (n % 10 === 7) return '#FFCC33'
  if (n % 10 === 8) return '#EF8200'
  if (n % 10 === 9) return '#E94C33'
}

export function render_segmentation (item) {
  if (!item.hasOwnProperty('class')) return
  const height = item.class.length
  const width = item.class[0].length
  const d = 1 // Resample drawing pixel.
  var canvas = new OffscreenCanvas(width / d, height / d)
  var cxt = canvas.getContext('2d')
  var imageData = cxt.getImageData(0, 0, width / d, height / d)
  cxt.clearRect(0, 0, width / d, height / d)

  for (let i = 0; i < height; i += d) {
    for (let j = 0; j < width; j += d) {
      const n = item.class[i][j]
      let c

      if (n % 10 === 0) c = 'E7009A'
      if (n % 10 === 1) c = '9F13C1'
      if (n % 10 === 2) c = '582396'
      if (n % 10 === 3) c = '0B20C4'
      if (n % 10 === 4) c = '3F9AAF'
      if (n % 10 === 5) c = '14884B'
      if (n % 10 === 6) c = 'BBAA19'
      if (n % 10 === 7) c = 'FFCC33'
      if (n % 10 === 8) c = 'EF8200'
      if (n % 10 === 9) c = 'E94C33'

      var bigint = parseInt(c, 16)
      var r = (bigint >> 16) & 255
      var g = (bigint >> 8) & 255
      var b = bigint & 255

      const img_index = (Math.floor(width * i / d / d) + Math.floor(j / d)) * 4
      imageData.data[img_index + 0] = r
      imageData.data[img_index + 1] = g
      imageData.data[img_index + 2] = b
      imageData.data[img_index + 3] = (n !== 0) * 160
    }
  }
  cxt.putImageData(imageData, 0, 0)
  return canvas.transferToImageBitmap()
}

export function setup_image_list (dataset, parent_width, parent_height, margin) {
  /**
    This
   */
  const brank = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
  const pages = []
  const img_list = dataset.img
  const size_list = dataset.size
  const last_index = img_list.length - 1

  let one_page = []
  let nth_line_in_page = 1
  let accumurated_ratio = 0
  const max_ratio = (parent_width / (parent_height / 3))

  for (let i = 0; i < size_list.length; i++) {
    const size = size_list[i]
    const ratio = ((size[0]) / (size[1]))
    accumurated_ratio += ratio

    if (accumurated_ratio < max_ratio || one_page.length === 0) {
      one_page.push({ index: i, img: img_list[i], size: size_list[i] })
    } else {
      if (nth_line_in_page >= 3) {
        pages.push(one_page)
        one_page = [{ index: i, img: img_list[i], size: size_list[i] }]
        accumurated_ratio = ratio
        nth_line_in_page = 1
      } else {
        one_page.push({ index: i, img: img_list[i], size: size_list[i] })
        accumurated_ratio = ratio
        nth_line_in_page++
      }
    }
    if (i === last_index) {
      // Add white image to empty space.
      one_page.push({ index: -1, img: brank, size: [max_ratio - accumurated_ratio, 1] })
      for (let j = nth_line_in_page; j <= 2; j++) {
        one_page.push({ index: -1, img: brank, size: [max_ratio, 1] })
      }
    }
  }
  if (pages[pages.length - 1] !== one_page) {
    pages.push(one_page)
  }
  return pages
}
