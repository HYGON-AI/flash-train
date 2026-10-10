# gfx936 grouped-kernel assembly

These four `.s` files are the authoritative sources for the packaged gfx936
native grouped-GEMM code objects.  They were taken from the current Flux tree.
The historical v36-named objects are not selected by the current resolver and
are not packaged by flash-train.

Source and packaged asset filenames use `hygon_`.  The ELF symbols inside the
assembly retain the historical `flux_` prefix because that prefix is the
existing loader ABI.  Run `make check` after changing an assembly source; it
fails unless the rebuilt object is byte-identical to the packaged asset.
