/**
 * Just enough WebGPU to probe for an adapter.
 *
 * @webgpu/types is thousands of lines describing an API this build does not call yet; when
 * Phase 2 lands and real pipelines exist, swap this file for the package.
 */

interface GPULike {
  requestAdapter(options?: unknown): Promise<unknown | null>;
}

interface Navigator {
  readonly gpu?: GPULike;
}

interface WorkerNavigator {
  readonly gpu?: GPULike;
}
