# ADR 0001: Import Flux distributed operators behind a Torch-only session adapter

## Status

Accepted

## Context

Flux contributes two production distributed operators for single-node expert
tensor parallelism:

- `AGGroupGemm`: all-gather plus grouped GEMM.
- `MoEReduceRS`: grouped GEMM followed by reduce-scatter.

Unlike the built-in local operators, both constructors require a PyTorch
ProcessGroup and allocate process-wide IPC buffers, private streams/events, and
route-specific device state.  Their execution path also loads architecture
specific code objects.  Consequently, they cannot be imported into
`libflash_train` without introducing Torch, ATen, `c10d`, and Python ABI
dependencies to the framework-neutral C library.

The operators are also session-oriented rather than stateless Pattern calls.
`AGGroupGemm` has five public route ABIs, and its input/output rows have
different distributed scope:

- input is rank-local,
- routing IDs describe the global token set,
- output contains the global expanded token set.

A local shape rule such as `output_rows == input_rows * topk` is therefore
incorrect for world sizes larger than one.  Route selection is part of the
session ABI, not a performance-only choice among interchangeable kernels.

## Decision

Import Flux as an optional backend of the PyTorch extension:

1. Keep `libflash_train` framework-neutral.  It must not include ATen, Torch
   distributed, Python headers, Flux sources, or Flux code objects.
2. Put the vendored Flux implementation and the concrete Torch session adapter
   under targets built only with `FTRAIN_BUILD_TORCH_EXTENSION=ON`.
3. Model the public operation in the framework-neutral Operation, Family, and
   Primitive layers.  A Primitive depends only on the abstract
   `DistributedSession` interface; it never includes ATen, Torch, or Python
   headers.  Avoid gratuitous namespace, environment, and kernel rewrites while
   establishing behavioral parity.
4. Package required gfx936/gfx938 code objects as Python package assets and
   configure explicit asset directories.  Installed wheels must not depend on
   the source checkout or process working directory.
5. Expose a direct compatibility API first.  It creates and invokes the Flux
   objects through the same ownership and stream rules as upstream Flux.
6. Register the formal flash-train engines with the other built-in engines.
   A core-only build can therefore describe the Pattern but returns
   unsupported when the concrete Torch session backend is absent.
7. Represent a distributed session by an opaque, never-reused integer ID in
   formal operation attributes.  `Problem` objects store that ID and tensor
   views, but do not own the session.  The backend primitive resolves and
   retains a session `shared_ptr` during `configure`.
8. Treat session route, variant, rank partition, capacity, dtype, and layout as
   immutable for formal Plans.  Mutable Flux sessions remain available through
   the direct compatibility API, but are not cacheable Plan state.
9. Use route-owned Primitives.  The Finder does not inspect global session
   state; applicability checks the session route and ABI after Problem
   construction.
10. Do not add a public distributed C API until the library has a
    framework-neutral communicator/session abstraction.  In particular, do not
    put `c10d::ProcessGroup` or ATen types in `include/flash_train/common.h`.

## Formal Plan semantics

The eventual formal Python API creates a collective session on every rank,
then creates a Plan with explicit input and output tensors.  Formal AG sessions
use `allocate_output_on_init=false`.  A Plan does not close or destroy its
session; closing a session only removes it from the named registry.  Existing
Plans retain their session through `shared_ptr` ownership.

AG Problem validation is based on distributed scope:

- input and topk IDs are rank-local/global views as defined by the session ABI,
- output width must match the weight output width,
- output rows describe the global expanded token set,
- route-specific local row relationships are checked by the backend Primitive.

MoEReduceRS validation similarly separates framework-independent tensor
relationships from route and weight-layout relationships owned by the session.

Primitives execute on the stream supplied by `ftrain::Resources`.  The formal
implementation wraps FTrain StorageViews as non-owning ATen tensors for every
invocation.  View caching is not part of the initial integration and may only
be added later with route-specific performance evidence and explicit view
ownership rules.

## Testing strategy

- Build and test the framework-neutral C library with Torch disabled.
- Build the wheel backend with Torch enabled.
- Import the installed wheel from outside the source tree and verify code object
  discovery.
- Exercise each AG route in a separate torchrun process because Flux route
  state and IPC resources are process-global.
- Compare direct compatibility results and performance with upstream Flux
  before enabling formal Plan selection.
- Add formal Plan correctness tests only after direct backend parity is
  established.

## Consequences

- The C library remains installable without Torch and continues to expose only
  framework-neutral types.
- Flux behavior is established before it is modeled by flash-train's engine.
- The first integration contains more backend code than a direct core-family
  import, but it keeps session lifetime, route ABI, and process-global state
  explicit.
- Public distributed C entrypoints are intentionally deferred.
