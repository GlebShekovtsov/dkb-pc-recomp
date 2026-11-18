// SPDX-License-Identifier: GPL-3.0-or-later
// The ASTC textures this renderer keeps as BC (TOTK_ASTC_BC) and the format their images are in: whether an upload
// counts the kinds of the blocks it writes, what the kinds say about the format (astc_opaque.h), the image made
// again in the format that calls for it and the entry left describing it. vk_astc.c encodes the BC1 blocks
// (bc1_encode.comp). The method is Hookshot's (gpu/vk/vk_texture.cpp bc_reformat, commit 97024ef5,
// GPL-3.0-or-later), written again in C for this renderer's images. The texture cache (vk_texture.c) owns the
// entries, the texture lock and the upload batches; it calls these and does its own bookkeeping after them.
#include "host.h"
#include "astc_opaque.h"
#include "vk.h"

// The entry's image made again as BC1 (to_bc1) or BC3 and the entry swapped over to it: the same extent, levels,
// layers, create flags and usage with the new format, in new memory, half or twice the video memory. The old
// image and the replay's views of it go once the GPU is done with them. The new image holds nothing and is in
// VK_IMAGE_LAYOUT_UNDEFINED, so its owner lays it out and fills it. The new image and its format go to img->image
// and img->format and through io_alloc, out_format and io_bc to the parts of the entry outside VkTexImage.
// 0 and nothing changed when the entry is not an ASTC image kept as BC (*io_bc 0), or the image cannot be made
// again: not a 2D single-sample image, one bound into pool memory, one with no memory of its own, or no memory
// for the new image.
static int vk_astc_bc_remake(VkTexImage *img, VkImageCreateFlags flags, VkImageUsageFlags usage, int srgb,
                             int to_bc1, VmaAllocation *io_alloc, VkFormat *out_format, uint8_t *io_bc) {
    static volatile int32_t n_to_bc1, n_to_bc3, n_failed;
    VkCtx *c = vk_ctx();
    if (!*io_bc || !c || img->type != VK_IMAGE_TYPE_2D || img->samples != 1) return 0;
    if (img->bound || !*io_alloc) return 0;   // pool memory: another image would alias the same pages
    VkFormat f = to_bc1 ? (srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK)
                        : (srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK);
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = img->type,
                            .format = f, .extent = {img->width, img->height, img->depth}, .mipLevels = img->levels,
                            .arrayLayers = img->layers, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage new_img;
    VmaAllocation new_alloc;
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ci, &ai, &new_img, &new_alloc, NULL) != VK_SUCCESS) {
        int32_t k = __atomic_add_fetch(&n_failed, 1, __ATOMIC_RELAXED);
        if (k <= 4)
            rt_log("vk: ASTC texture %#llx %ux%u: its image could not be made again as %s; it stays as it is",
                   (unsigned long long)img->addr, img->width, img->height, to_bc1 ? "BC1" : "BC3");
        return 0;
    }
    // The old image and the replay's views of it go once the GPU is done with them. The new one is the entry's from
    // here on, so it is not deferred; it holds nothing and is still UNDEFINED, for the owner to lay out.
    vk_exec_image_gone(img->image);
    vk_defer_image(img->image, *io_alloc);
    img->image = new_img, img->format = f;
    *io_alloc = new_alloc, *out_format = f, *io_bc = (uint8_t)(to_bc1 ? 2 : 1);
    int32_t k = __atomic_add_fetch(to_bc1 ? &n_to_bc1 : &n_to_bc3, 1, __ATOMIC_RELAXED);
    if (k <= 4 || (k & (k - 1)) == 0)
        rt_log("vk: ASTC texture %#llx %ux%u x%u: %s (%d opaque textures kept as BC1, %d back to BC3; "
               "TOTK_ASTC_BC1=0: all BC3)", (unsigned long long)img->addr, img->width, img->height, img->layers,
               to_bc1 ? "every block opaque, kept as BC1" : "alpha in it now, BC3 again",
               (int)__atomic_load_n(&n_to_bc1, __ATOMIC_RELAXED), (int)__atomic_load_n(&n_to_bc3, __ATOMIC_RELAXED));
    return 1;
}

// Whether an upload counts the kinds of the blocks it is about to write: an ASTC image kept as BC (astc_bc 1 =
// BC3, 2 = BC1) whose raw blocks go to the GPU as they are (agpu), that is mirrored in memory of its own (bound
// into no pool, alloc set), and not once it is to stay in the format it has (bc_settled). The kinds are counted
// from the deswizzled blocks in that memory, not read back out of the staging buffer (write-combined memory).
int vk_astc_bc_kinds(int agpu, uint8_t astc_bc, const VkTexImage *img, VmaAllocation alloc,
                     VmaAllocation bound_to, uint8_t bc_settled) {
    return agpu && astc_bc && vk_astc_bc1() && img->mirrored && !bc_settled && !bound_to && alloc;
}

// The 16-byte blocks of [p, p + n) counted by kind and added to *s (astc_opaque.h). The count stops at the first
// block that may carry alpha: that is all the choice needs to know, and a texture still being filled with zeros
// is counted whole up to there and no further.
void vk_astc_bc_scan(AstcOpaqueScan *s, const uint8_t *p, uint64_t n) {
    if (!s->alpha) astc_opaque_scan(s, p, n, 1);
}

// What an upload does with the kinds it counted (astc_bc_choice, astc_bc_settled): the entry's image made again in
// the other format when they call for it and the entry left describing it, and bc_settled set on the entry when
// the texture is to stay as it is for good and its blocks are not counted again. 1 when the image was made
// again, so its owner transitions it and uploads every level and layer into it.
int vk_astc_bc_choice(VkTexImage *img, VkImageCreateFlags flags, VkImageUsageFlags usage, int srgb,
                      const AstcOpaqueScan *s, VmaAllocation *io_alloc, VkFormat *out_format, uint8_t *io_bc,
                      uint8_t *io_settled) {
    int was = *io_bc, choice = astc_bc_choice(was, s), remade = 0;
    if (choice != ASTC_BC_KEEP) remade = vk_astc_bc_remake(img, flags, usage, srgb, choice == ASTC_BC_TO_BC1,
                                                     io_alloc, out_format, io_bc);
    if (astc_bc_settled(was, choice, remade, s)) *io_settled = 1;
    return remade;
}

// An entry whose ASTC image is kept as BC1 (io_bc 2) has it made BC3 again and the entry left describing the
// new image; every other image is left as it is. 1 when the image is BC3 and the caller may go on with it, 0
// when it is a BC1 image whose remake failed: a BC1 block is 8 bytes and holds no alpha for the texels the
// caller writes. bc_settled is set on the entry when the texture is to stay BC3 for good and its blocks are not
// counted again.
int vk_astc_bc3(VkTexImage *img, VkImageCreateFlags flags, VkImageUsageFlags usage, int srgb,
                VmaAllocation *io_alloc, VkFormat *out_format, uint8_t *io_bc, uint8_t *io_settled) {
    if (*io_bc != 2) return 1;   // not an ASTC image kept as BC1: it is already what the caller wants
    if (!vk_astc_bc_remake(img, flags, usage, srgb, 0, io_alloc, out_format, io_bc)) return 0;
    *io_settled = 1;
    return 1;
}
