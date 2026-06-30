package hnsw

// Header offsets for the 128-byte file header (version 6 layout).
// All values are uint32 little-endian at the given byte offsets.
const (
	hdrMagic          = 0
	hdrVersion        = 4
	hdrDims           = 8
	hdrM              = 12
	hdrMMax0          = 16
	hdrMaxLevel       = 20
	hdrEntryPoint     = 24
	hdrNodeCount      = 28
	hdrAllocated      = 32
	hdrMaxLevelDyn    = 36
	hdrProbes         = 40
	hdrEfSearch       = 44
	hdrEfConst        = 48
	hdrDeletedCount   = 52
	hdrUpperUsed      = 56
	hdrUpperAllocated = 60
	hdrMetaUsed       = 64
	hdrMetaAllocated  = 68
)
