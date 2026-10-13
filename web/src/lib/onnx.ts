/**
 * The one seam between this app and an ONNX runtime. The browser uses
 * onnxruntime-web (WebAssembly, single thread, no cross-origin isolation
 * needed). Tests pass a runner built on onnxruntime-node so the very same
 * model files are exercised without a browser.
 */

export interface OutputTensor {
  data: ArrayLike<number> | ArrayLike<bigint>
  dims: readonly number[]
}

export interface Runner {
  run(input: Float32Array, dims: number[]): Promise<Record<string, OutputTensor>>
}

export type RunnerFactory = (model: string) => Promise<Runner>

const cache = new Map<string, Promise<Runner>>()

/** Loads a model from a URL once and reuses the session. */
export const webRunner: RunnerFactory = (url) => {
  let pending = cache.get(url)
  if (!pending) {
    pending = (async () => {
      const ort = await import('onnxruntime-web/wasm')
      ort.env.wasm.wasmPaths = '/ort/'
      ort.env.wasm.numThreads = 1
      const session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
      const inputName = session.inputNames[0]!
      return {
        async run(input: Float32Array, dims: number[]) {
          const out = await session.run({ [inputName]: new ort.Tensor('float32', input, dims) })
          return out as unknown as Record<string, OutputTensor>
        },
      }
    })()
    cache.set(url, pending)
    pending.catch(() => cache.delete(url))
  }
  return pending
}
