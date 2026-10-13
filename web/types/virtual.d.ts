declare module 'virtual:art' {
  const art: Record<string, { url: string; source: 'generated' | 'interim' }>
  export default art
}

declare module 'gifenc' {
  const gifenc: {
    GIFEncoder: () => {
      writeFrame: (index: Uint8Array, width: number, height: number, options: Record<string, unknown>) => void
      finish: () => void
      bytes: () => Uint8Array
    }
    quantize: (rgba: Uint8Array, maxColors: number) => number[][]
    applyPalette: (rgba: Uint8Array, palette: number[][]) => Uint8Array
  }
  export default gifenc
}
