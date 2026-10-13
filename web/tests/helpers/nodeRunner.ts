import path from 'node:path'
import * as ort from 'onnxruntime-node'
import type { OutputTensor, Runner } from '../../src/lib/onnx'

export const MODELS = path.resolve(__dirname, '..', '..', 'public', 'models')
export const REPO = path.resolve(__dirname, '..', '..', '..')

/** The same Runner interface the browser uses, backed by onnxruntime-node. */
export async function nodeRunner(file: string): Promise<Runner> {
  const session = await ort.InferenceSession.create(path.join(MODELS, file))
  const inputName = session.inputNames[0]!
  return {
    async run(input: Float32Array, dims: number[]) {
      const out = await session.run({ [inputName]: new ort.Tensor('float32', input, dims) })
      return out as unknown as Record<string, OutputTensor>
    },
  }
}
