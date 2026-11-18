// Two-pane settings pages (option_rows.h): rows on the left, the selected or hovered row's description on the right.
//
// Adapted from SavageWindRecomp's runtime/ui/option_rows.mm (https://github.com/Revivedx/SavageWindRecomp @
// 575ad88e02f9a89a9a885d90039d48af33c982f4, MIT, Copyright (c) 2026 Revivedx and the Savage Wind Recomp contributors,
// Copyright (c) 2026 Peiikai): the row look, the two panes, the left / right stepping and the hold speed-up. The rows
// read and write cfg keys instead of a settings struct, take their text from the menu catalog (menu_text), and show
// the default, restart and environment notes; the text field, radio and on-screen keyboard rows are not ported.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imgui.h"
#include "imgui_internal.h"   // SetKeyOwner, SetFocusID, NavCursorVisible

#include "ui/option_rows.h"

extern "C" {
#include "cfg.h"
#include "platform/platform.h"
#include "ui_text.h"
// host.h pulls in cpu.h, which is C only: the one function needed from it, as declared there.
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
}

// The colours of the rows. "classic" is the SavageWindRecomp look the menu started with; "dk" is the jungle-bark and
// banana palette (menu.cpp style_init has the window colours of both). opt_theme picks one.
struct Pal {
    bool dk;
    ImU32 gold, dim, grey;
    ImU32 fill_idle, fill_on;                            // row background: idle, hovered or selected
    ImU32 border_idle, border_idle_off, border_on_off;   // row border: idle, idle when greyed out, hovered when greyed out
    ImU32 arrows_idle, divider;
    ImU32 label, value_default, value_set, off_note, set_note;
    ImU32 cell_text;
    ImVec4 accent, warn;
};
static const Pal kClassic = {false, IM_COL32(236, 194, 84, 255), IM_COL32(150, 150, 140, 255), IM_COL32(110, 110, 102, 255),
                             IM_COL32(0, 0, 0, 40), IM_COL32(255, 255, 255, 22),
                             IM_COL32(255, 255, 255, 38), IM_COL32(255, 255, 255, 18), IM_COL32(236, 194, 84, 120),
                             IM_COL32(170, 170, 160, 255), IM_COL32(236, 194, 84, 110),
                             IM_COL32(240, 236, 224, 255), IM_COL32(205, 205, 195, 255), IM_COL32(255, 255, 255, 255),
                             IM_COL32(200, 150, 90, 255), IM_COL32(220, 170, 100, 255),
                             IM_COL32(255, 255, 255, 255),
                             ImVec4(0.93f, 0.76f, 0.33f, 1), ImVec4(0.95f, 0.66f, 0.40f, 1)};
static const Pal kDk = {true, IM_COL32(246, 197, 43, 255), IM_COL32(190, 168, 138, 255), IM_COL32(140, 120, 98, 255),
                        IM_COL32(0, 0, 0, 70), IM_COL32(246, 197, 43, 34),
                        IM_COL32(107, 74, 42, 255), IM_COL32(107, 74, 42, 110), IM_COL32(246, 197, 43, 120),
                        IM_COL32(190, 168, 138, 255), IM_COL32(107, 74, 42, 255),
                        IM_COL32(243, 233, 210, 255), IM_COL32(231, 214, 184, 255), IM_COL32(255, 244, 214, 255),
                        IM_COL32(214, 150, 90, 255), IM_COL32(240, 190, 100, 255),
                        IM_COL32(255, 244, 214, 255),
                        ImVec4(246 / 255.f, 197 / 255.f, 43 / 255.f, 1), ImVec4(1.0f, 0.54f, 0.46f, 1)};
static const Pal *P = &kClassic;
#define kGold (P->gold)
#define kDim (P->dim)
#define kGrey (P->grey)

void opt_theme(bool dk) { P = dk ? &kDk : &kClassic; }
ImVec4 opt_accent(void) { return P->accent; }
ImVec4 opt_warn(void) { return P->warn; }

static float g_scale = 1;
static bool g_focus_req, g_take_first;   // opt_focus_first: requested / this page's first row takes it
static char g_focus_key[64];             // opt_focus_key: the row that takes the cursor when it is next drawn
static bool g_held;                      // the last row's left / right has been held a while (steppers go faster)
static float g_bottom;                   // opt_reserve_bottom: kept free under the panes
static bool g_disable_next;              // opt_disable_next: the next row is greyed out with g_disable_note
static UiTextId g_disable_note;

// Help belongs to the current page. Copy status text because callers can reuse their formatting buffers.
struct Desc {
    ImGuiID row;
    const char *title, *body;
    char requirement[160], status[160];
    bool restart, is_default;
};
static Desc g_hover, g_focus, g_previous, g_first;
static ImGuiID g_last_help;

// disabled: the note of a row opt_disable_next greyed out, else NULL.
static void offer_help(ImGuiID row, bool hovered, bool focused, const char *title, UiTextId desc,
                       const char *key, const CfgMeta *m, const char *disabled,
                       bool is_default = false, const char *status = nullptr) {
    bool previous = row == g_last_help;
    if (!hovered && !focused && !previous && g_first.row) return;
    Desc d = {};
    d.row = row;
    d.title = title;
    d.body = menu_text(desc);
    bool from_env = cfg_from_env(key) != 0;
    d.restart = !disabled && !from_env && m && !m->live;
    d.is_default = is_default && !disabled && !from_env;
    if (disabled)
        snprintf(d.requirement, sizeof d.requirement, "%s", disabled);
    else if (from_env)
        snprintf(d.requirement, sizeof d.requirement, "%s %s", menu_text(UI_MENU_SET_BY_ENV), key);
    if (status) snprintf(d.status, sizeof d.status, "%s", status);
    if (hovered) g_hover = d;
    if (focused) g_focus = d;
    if (previous) g_previous = d;
    if (!g_first.row) g_first = d;
}

void opt_focus_first(void) { g_focus_req = true; }

void opt_focus_first_item(void) {
    if (!g_take_first) return;
    ImGui::SetFocusID(ImGui::GetItemID(), ImGui::GetCurrentWindow());
    ImGui::SetNavCursorVisible(true);
    ImGui::SetScrollY(0);
    g_take_first = false;
}

void opt_focus_key(const char *key) { snprintf(g_focus_key, sizeof g_focus_key, "%s", key); }

void opt_reserve_bottom(float h) { g_bottom = h; }

void opt_disable_next(UiTextId note) {
    g_disable_next = true;
    g_disable_note = note;
}

void opt_begin(const char *id, float scale) {
    g_scale = scale;
    g_take_first = g_focus_req;
    g_focus_req = false;
    if (g_take_first) ImGui::NavMoveRequestCancel();   // this frame's move goes to the first row instead
    g_hover.row = g_focus.row = g_previous.row = g_first.row = 0;
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float rows_height = ImMax(1.0f, avail.y - g_bottom - 18 * scale);
    ImGui::PushID(id);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14 * scale, 8 * scale));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, P->dk ? IM_COL32(35, 24, 16, 255) : IM_COL32(22, 20, 17, 255));
    ImGui::BeginChild("##rows", ImVec2(avail.x * 0.58f, rows_height),
                      ImGuiChildFlags_NavFlattened | ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 5 * scale));
}

void opt_end(void) {
    bool more_below = ImGui::GetScrollY() + 1 < ImGui::GetScrollMaxY();
    ImGui::PopStyleVar();
    ImGui::EndChild();
    if (more_below) {
        ImVec2 left = ImGui::GetItemRectMin(), right = ImGui::GetItemRectMax();
        float x = (left.x + right.x) * 0.5f, y = right.y + 8 * g_scale;
        float dx = 5 * g_scale, dy = 2.5f * g_scale;
        ImDrawList *draw = ImGui::GetWindowDrawList();
        draw->AddLine(ImVec2(x - dx, y - dy), ImVec2(x, y + dy), kGold, 2 * g_scale);
        draw->AddLine(ImVec2(x, y + dy), ImVec2(x + dx, y - dy), kGold, 2 * g_scale);
    }
    // Keep help readable when the pointer leaves the row, as long as that row is still on the page.
    const Desc &shown = g_hover.row ? g_hover : g_focus.row ? g_focus : g_previous.row ? g_previous : g_first;
    g_last_help = shown.row;
    ImGui::SameLine(0, 0);
    ImVec2 p = ImGui::GetCursorScreenPos(), avail = ImGui::GetContentRegionAvail();
    avail.y -= g_bottom;
    float gap = 24 * g_scale;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x + gap * 0.5f, p.y), ImVec2(p.x + gap * 0.5f, p.y + avail.y),
                                        P->divider, P->dk ? 2.0f : 1.0f);
    ImGui::SetCursorScreenPos(ImVec2(p.x + gap, p.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14 * g_scale, 6 * g_scale));
    ImGui::BeginChild("##desc", ImVec2(0, avail.y), ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoNav);
    ImGui::PopStyleVar();
    if (shown.row) {
        ImGui::PushTextWrapPos(0);
        ImGui::TextColored(opt_accent(), "%s", shown.title);
        ImGui::Spacing();
        ImGui::TextUnformatted(shown.body);
        if (shown.restart || shown.requirement[0] || shown.status[0] || shown.is_default) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            const ImVec4 amber = ImGui::ColorConvertU32ToFloat4(P->set_note);
            if (shown.restart) ImGui::TextColored(amber, "%s", menu_text(UI_MENU_AFTER_RESTART));
            if (shown.requirement[0]) ImGui::TextColored(amber, "%s", shown.requirement);
            if (shown.status[0]) ImGui::TextColored(amber, "%s", shown.status);
            if (shown.is_default) ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kDim), "%s",
                                                    menu_text(UI_MENU_IS_DEFAULT));
        }
        ImGui::PopTextWrapPos();
    }
    ImGui::EndChild();
    ImGui::PopID();
}

// What a row shows besides its label.
struct RowView {
    const char *value;   // right side; "" for none
    bool is_default;     // the value is the setting's default; shown in the description pane
    bool stepper;        // a number with fixed left/right buttons
    const char *note;    // status under the label, e.g. the actual frame sync mode; NULL for none
    int pill = 0;        // dk theme: the value sits in a green (1) or red (2) pill, a toggle's on / off
    float frac = -1;     // dk theme: a slider's position 0..1 drawn as a track with a grab; -1 none
    const char *status = nullptr;   // optional longer description-panel status; note remains concise on the row
};

struct ChoiceList {
    const char *const *names;
    int count, current;
    int picked = -1;
};

// Keep one navigation item for the whole row, but anchor ImGui's combo popup to the value field.
static void choice_popup(ChoiceList &choices, const ImRect &field, bool activate) {
    const char *id = "##choices";
    if (activate) {
        ImGui::OpenPopup(id);
        ImGui::SetNextWindowPos(field.GetBL(), ImGuiCond_Appearing);
    }
    if (!ImGui::IsPopupOpen(id)) return;
    // ImGui uses the anchor width as a minimum. Anchor to the left edge so the list can fit its choices.
    ImRect anchor(field.Min, field.GetBL());
    ImGui::PushStyleVarX(ImGuiStyleVar_FramePadding, 10 * g_scale);
    ImGui::PushStyleVarY(ImGuiStyleVar_WindowPadding, 6 * g_scale);
    bool open = ImGui::BeginComboPopup(ImGui::GetID(id), anchor, ImGuiComboFlags_None);
    ImGui::PopStyleVar(2);
    if (open) {
        for (int i = 0; i < choices.count; ++i) {
            ImGui::PushID(i);
            bool current = i == choices.current;
            if (ImGui::Selectable(choices.names[i], current)) choices.picked = i;
            if (current) ImGui::SetItemDefaultFocus();
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
}

// One row: draws it, handles A / click and left / right while selected. Returns 2 (A / click), -1 / +1 (left / right) or
// 0; always 0 for a row the environment fixes (greyed out, still selectable so its description shows).
static int row(const char *key, UiTextId label, UiTextId desc, const RowView &v, ChoiceList *choices = nullptr) {
    const CfgMeta *m = cfg_meta(key);
    const char *disabled = g_disable_next ? menu_text(g_disable_note) : NULL;
    g_disable_next = false;
    bool off = disabled || cfg_from_env(key) != 0;
    ImGui::PushID(key);
    ImVec2 p = ImGui::GetCursorScreenPos();
    // h whole pixels: ImGui truncates cursor positions, which rounds a fractional height the other way above the window's
    // top (negative y), so the page grew a pixel per row scrolled out and a row scrolled to the bottom was cut off
    float fs = ImGui::GetFontSize(), w = ImGui::GetContentRegionAvail().x;
    bool has_track = P->dk && v.frac >= 0;
    float field_height = fs * 1.33f;
    float h = floorf(has_track ? fmaxf(fs * 2.2f, field_height + 26 * g_scale) : fs * 2.2f);
    ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4(0, 0, 0, 0));   // the row's own border shows the selection
    bool clicked = ImGui::InvisibleButton("##row", ImVec2(w, h), ImGuiButtonFlags_EnableNav);
    ImGui::PopStyleColor();
    bool to_key = g_focus_key[0] && !strcmp(key, g_focus_key);
    if (g_take_first || to_key) {
        ImGui::SetFocusID(ImGui::GetItemID(), ImGui::GetCurrentWindow());
        ImGui::SetNavCursorVisible(true);
        if (g_take_first) ImGui::SetScrollY(0);   // the page's top: text and a heading above the first row show
        else ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleEdgeY);   // as a cursor move would (the last row whole)
        g_take_first = false;
        if (to_key) g_focus_key[0] = 0;
    }
    bool hov = ImGui::IsItemHovered(), foc = ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
    int step = clicked && !off ? 2 : 0;
    g_held = false;
    if (foc && !off) {
        // the left stick arrives as the D-pad (menu.cpp pad_events)
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft) || ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) step = -1;
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight) || ImGui::IsKeyPressed(ImGuiKey_RightArrow)) step = 1;
        static const ImGuiKey steps[] = {ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadDpadRight, ImGuiKey_LeftArrow,
                                         ImGuiKey_RightArrow};
        for (ImGuiKey k : steps) {
            if (ImGui::GetKeyData(k)->DownDuration > 1.0f) g_held = true;
            // left and right belong to the row (navigation would otherwise move sideways into the description pane)
            ImGui::SetKeyOwner(k, ImGui::GetItemID());
        }
    }
    const char *title = menu_text(label);
    offer_help(ImGui::GetItemID(), hov, foc, title, desc, key, m, disabled, v.is_default, v.status ? v.status : v.note);

    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImFont *font = ImGui::GetFont();
    float r = (P->dk ? 6 : 9) * g_scale, pad = 16 * g_scale, small = fs * 0.72f;
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), hov || foc ? P->fill_on : P->fill_idle, r);
    dl->AddRect(p, ImVec2(p.x + w, p.y + h),
                hov || foc ? (off ? P->border_on_off : kGold) : off ? P->border_idle_off : P->border_idle, r, 0,
                hov || foc ? (P->dk ? 2.5f * g_scale : 2.0f) : P->dk ? 1.5f * g_scale : 1.0f);
    if (P->dk) {   // the plank's bevel: a light edge on top, a dark one under
        dl->AddLine(ImVec2(p.x + r, p.y + 2 * g_scale), ImVec2(p.x + w - r, p.y + 2 * g_scale), IM_COL32(255, 255, 255, 16));
        dl->AddLine(ImVec2(p.x + r, p.y + h - 2 * g_scale), ImVec2(p.x + w - r, p.y + h - 2 * g_scale),
                    IM_COL32(0, 0, 0, 70));
    }
    const char *note = v.note;
    ImU32 note_color = P->set_note;
    if (off) {
        note = disabled ? disabled : menu_text(UI_MENU_SET_BY_ENV);
        note_color = P->off_note;
    } else if (m && !m->live) {
        note = menu_text(UI_MENU_AFTER_RESTART);
        note_color = P->set_note;
    }

    float value_y = p.y + (h - fs) * 0.5f;
    float gap = 2 * g_scale;
    float ty = note ? value_y - (small + gap) * 0.5f : value_y;
    float ny = ty + fs + gap;
    float right = p.x + w - pad;                         // the value's right edge
    float value_left = right;
    ImU32 vc = off ? kGrey : v.is_default ? P->value_default : P->value_set;
    if (choices || v.stepper) {
        float width = floorf(fminf(300 * g_scale, w * 0.47f));
        value_left = right - width;
        float y0 = p.y + (h - field_height) * 0.5f;
        float y1 = y0 + field_height, cy = y0 + field_height * 0.5f;
        ImU32 ac = off ? kGrey : hov || foc ? kGold : P->arrows_idle;
        ImU32 bg = ImGui::GetColorU32(ImGuiCol_FrameBg, off ? 0.55f : 1.0f);
        ImU32 hover_bg = ImGui::GetColorU32(ImGuiCol_FrameBgHovered);
        float round = 4 * g_scale;
        if (choices) {
            ImRect field(ImVec2(value_left, y0), ImVec2(right, y1));
            dl->AddRectFilled(field.Min, field.Max, !off && (hov || foc) ? hover_bg : bg, round);
            dl->AddRect(field.Min, field.Max, off ? P->border_idle_off : P->border_idle, round);
            ImGui::RenderArrow(dl, ImVec2(right - fs * 1.13f, cy - fs * 0.35f), ac, ImGuiDir_Down, 0.7f);
            ImVec2 start(value_left + fs * 0.42f, cy - fs * 0.5f);
            ImVec2 end(right - fs * 1.15f, y1);
            if (end.x > start.x) {
                ImGui::PushStyleColor(ImGuiCol_Text, vc);
                ImGui::RenderTextEllipsis(dl, start, end, end.x, end.x, v.value, nullptr, nullptr);
                ImGui::PopStyleColor();
            }
            if (!off && choices->count > 0) choice_popup(*choices, field, step == 2);
        } else {
            float button_w = fminf(fs * 1.5f, width * 0.28f);
            float xs[] = {value_left, right - button_w};
            float mx = ImGui::GetIO().MousePos.x;
            bool mouse_click = step == 2 && ImGui::IsMouseReleased(ImGuiMouseButton_Left);
            if (mouse_click) step = 0;
            for (int i = 0; i < 2; ++i) {
                float x = xs[i];
                bool hot = !off && hov && mx >= x && mx < x + button_w;
                dl->AddRectFilled(ImVec2(x, y0), ImVec2(x + button_w, y1), hot ? hover_bg : bg, round);
                ImGui::RenderArrow(dl, ImVec2(x + (button_w - fs) * 0.5f, cy - fs * 0.35f), ac,
                                   i == 0 ? ImGuiDir_Left : ImGuiDir_Right, 0.7f);
                if (mouse_click && hot) step = i == 0 ? -1 : 1;
            }
            float x = (value_left + right - ImGui::CalcTextSize(v.value).x) * 0.5f;
            ImVec4 clip(value_left + button_w, y0, right - button_w, y1);
            dl->AddText(font, fs, ImVec2(x, cy - fs * 0.5f), vc, v.value, nullptr, 0, &clip);
        }
    } else if (v.value[0]) {
        float vx = right - ImGui::CalcTextSize(v.value).x;
        if (P->dk && v.pill) {   // green for on, red for off; the text stays white for contrast
            float px = fs * 0.5f;
            vx -= px;
            dl->AddRectFilled(ImVec2(vx - px, value_y + fs * 0.02f), ImVec2(right, value_y + fs * 0.98f),
                              v.pill == 1 ? IM_COL32(46, 125, 43, off ? 120 : 255) : IM_COL32(168, 42, 32, off ? 120 : 255),
                              fs * 0.5f);
            vc = off ? kGrey : IM_COL32(255, 255, 255, 255);
            value_left = vx - px;
        }
        dl->AddText(ImVec2(vx, value_y), vc, v.value);
        if (!v.pill) value_left = vx;
    }
    // the label and its note, clipped before the value
    ImVec4 clip(p.x, p.y, value_left - fs * 0.6f, p.y + h);
    dl->AddText(font, fs, ImVec2(p.x + pad, ty), off ? kGrey : P->label, title, NULL, 0,
                &clip);
    if (note) dl->AddText(font, small, ImVec2(p.x + pad, ny), note_color, note, nullptr, 0, &clip);
    if (has_track) {   // the slider's track with a banana grab, along the row's foot
        float x0 = p.x + pad, x1 = p.x + w - pad, y = p.y + h - 5 * g_scale, gx = x0 + (x1 - x0) * v.frac;
        dl->AddLine(ImVec2(x0, y), ImVec2(x1, y), IM_COL32(0, 0, 0, 140), 3 * g_scale);
        dl->AddLine(ImVec2(x0, y), ImVec2(gx, y), off ? kGrey : kGold, 3 * g_scale);
        dl->AddCircleFilled(ImVec2(gx, y), 5.5f * g_scale, off ? kGrey : kGold);
        dl->AddCircle(ImVec2(gx, y), 5.5f * g_scale, IM_COL32(42, 26, 14, 255), 0, 1.5f * g_scale);
    }
    ImGui::PopID();
    return off ? 0 : step;
}

// ---- writing a setting: at once in memory (live settings follow), to totk.ini a moment later (opt_save_tick)
static uint64_t g_save_at;   // plat_now_ns deadline of the pending write; 0 none
static bool g_save_logged;

static void apply(const char *key, const char *value) {
    if (*value) {
        cfg_set(key, value);
    } else {   // the automatic choice: the key unset, its line removed from totk.ini
        const char *k = key;
        cfg_reset(&k, 1);
    }
    opt_save_soon();
}

void opt_save_soon(void) { g_save_at = plat_now_ns() + 400000000ull; }

int opt_save_tick(int force) {
    if (!g_save_at || (!force && plat_now_ns() < g_save_at)) return g_save_at != 0;
    g_save_at = 0;
    if (!cfg_save() && !g_save_logged) {
        g_save_logged = true;
        rt_log("menu: could not write totk.ini; the changed settings hold for this session");
    }
    return 0;
}

// ---- rows

bool opt_toggle(const char *key, UiTextId label, UiTextId desc, bool inverted) {
    const CfgMeta *m = cfg_meta(key);
    bool set = cfg_bool(key) != 0, def = m && strcmp(m->def, "0") != 0;
    bool on = set != inverted;   // what the row shows
    RowView v = {menu_text(on ? UI_MENU_ON : UI_MENU_OFF), set == def, false, NULL};
    v.pill = on ? 1 : 2;
    if (!row(key, label, desc, v)) return false;   // any step flips it
    apply(key, set ? "0" : "1");
    return true;
}

bool opt_button(const char *id, UiTextId label, UiTextId desc, const char *value) {
    RowView v = {value ? value : "", false, false, NULL};
    return row(id, label, desc, v) == 2;   // A or a click; left and right do nothing
}

void opt_group(UiTextId title) {
    ImGui::Dummy(ImVec2(0, 6 * g_scale));
    ImGui::TextColored(P->accent, "%s", menu_text(title));
}

void opt_text(UiTextId text, bool warn) {
    ImGui::PushTextWrapPos(0);
    if (warn) ImGui::TextColored(P->warn, "%s", menu_text(text));
    else ImGui::TextUnformatted(menu_text(text));
    ImGui::PopTextWrapPos();
}

bool opt_choice_list(const char *key, UiTextId label, const char *const *values, const char *const *names, int n,
                     UiTextId desc, int cur, const char *note, const char *status) {
    const CfgMeta *m = cfg_meta(key);
    const char *now = cfg_value(key);
    if (cur < 0)
        for (int i = 0; i < n; i++)
            if (!strcmp(values[i], now)) cur = i;
    int def = -1;
    for (int i = 0; m && i < n; i++)
        if (!strcmp(values[i], m->def)) def = i;
    if (cur >= n) cur = -1;
    RowView v = {cur >= 0 ? names[cur] : now, cur >= 0 && cur == def, false, note};
    v.status = status;
    ChoiceList choices = {names, n, cur};
    int step = row(key, label, desc, v, &choices);
    int nv = choices.picked;
    if (nv < 0 && n > 0 && (step == -1 || step == 1))
        nv = cur < 0 ? 0 : ImClamp(cur + step, 0, n - 1);   // left/right still work without opening the list
    if (nv < 0 || nv == cur) return false;
    apply(key, values[nv]);
    return true;
}

bool opt_choice(const char *key, UiTextId label, const UiTextId *names, int n, UiTextId desc, const char *note) {
    enum { MAX = 32 };
    const CfgMeta *m = cfg_meta(key);
    const char *values[MAX], *shown[MAX];
    char buf[256];
    int k = 0;
    if (m && !m->def[0]) values[k++] = "";   // the automatic choice
    snprintf(buf, sizeof buf, "%s", m && m->values ? m->values : "");
    for (char *s = buf, *e; *s && k < MAX; s = e) {
        e = strchr(s, '|');
        if (e) *e++ = 0;
        else e = s + strlen(s);
        values[k++] = s;
    }
    if (n > k) n = k;
    for (int i = 0; i < n; i++) shown[i] = menu_text(names[i]);
    return opt_choice_list(key, label, values, shown, n, desc, -1, note);
}

bool opt_slider(const char *key, UiTextId label, float lo, float hi, UiTextId desc, float step, const char *fmt,
                const char *note) {
    const CfgMeta *m = cfg_meta(key);
    double v = cfg_double(key), def = m ? strtod(m->def, NULL) : lo;
    char shown[48];
    snprintf(shown, sizeof shown, fmt, v);
    RowView rv = {shown, fabs(v - def) < step * 0.01, true, note};
    rv.frac = hi > lo ? (float)fmin(1.0, fmax(0.0, (v - lo) / (hi - lo))) : 0;
    int s = row(key, label, desc, rv);
    if (!s) return false;
    int steps = g_held && (hi - lo) > 40 * step ? 10 : 1;   // held: ten steps at a time over a long range
    double n = round(v / step) * step + (s == 2 ? step : s * step * steps);
    if (s == 2 && n > hi + step * 0.01) n = lo;   // A / click past the top goes round; left / right stop at the ends
    n = n < lo ? lo : n > hi ? hi : n;
    if (fabs(n - v) < step * 0.01) return false;
    char out[32];
    if (step >= 1) snprintf(out, sizeof out, "%d", (int)lround(n));
    else snprintf(out, sizeof out, "%g", round(n / step) * step);
    apply(key, out);
    return true;
}

// ---- binding rows: one controller cell or two key cells, each its own nav item

// Text centred (left-aligned when centred is false) in [x0, x1] on a row from y to y + h, at the font size or smaller so
// a long name ("Нажатие правого стика", "Стрелка вправо") fits; under 65 % it is clipped instead.
static void fit_text(ImDrawList *dl, float x0, float x1, float y, float h, ImU32 col, const char *t, bool centred) {
    ImFont *font = ImGui::GetFont();
    float fs = ImGui::GetFontSize(), room = x1 - x0, tw = font->CalcTextSizeA(fs, FLT_MAX, 0, t).x;
    if (tw > room) {
        fs = fmaxf(fs * room / tw, fs * 0.65f);
        tw = font->CalcTextSizeA(fs, FLT_MAX, 0, t).x;
    }
    float x = centred && tw < room ? x0 + (room - tw) * 0.5f : x0;
    ImVec4 clip(x0, y, x1, y + h);
    dl->AddText(font, fs, ImVec2(x, y + (h - fs) * 0.5f), col, t, NULL, 0, &clip);
}

static void bind_columns(float x, float w, int count, float cx[2], float *cw) {
    float lw = floorf(w * 0.42f), gap = floorf(8 * g_scale);
    *cw = floorf((w - lw - gap * count) / count);
    for (int c = 0; c < count; c++) cx[c] = x + lw + gap + c * (*cw + gap);
}

void opt_bind_key_header(void) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x, cx[2], cw, fs = ImGui::GetFontSize() * 0.8f;
    bind_columns(p.x, w, 2, cx, &cw);
    const UiTextId cols[] = {UI_BINDS_COL_KEY1, UI_BINDS_COL_KEY2};
    ImFont *font = ImGui::GetFont();
    for (int c = 0; c < 2; c++) {
        const char *t = menu_text(cols[c]);
        float tw = font->CalcTextSizeA(fs, FLT_MAX, 0, t).x;
        ImGui::GetWindowDrawList()->AddText(font, fs, ImVec2(cx[c] + (cw - tw) * 0.5f, p.y), kDim, t);
    }
    ImGui::Dummy(ImVec2(w, fs * 1.2f));
}

int opt_bind_row(const char *key, const char *label, UiTextId desc, const char *const cells[3], const bool enabled[3], bool controller,
                 const char *note) {
    ImGui::PushID(key);
    ImVec2 p = ImGui::GetCursorScreenPos();
    float fs = ImGui::GetFontSize(), w = ImGui::GetContentRegionAvail().x, h = floorf(fs * 1.9f), cx[2], cw;
    int count = controller ? 1 : 2;
    bind_columns(p.x, w, count, cx, &cw);
    ImDrawList *dl = ImGui::GetWindowDrawList();
    float r = (P->dk ? 6 : 8) * g_scale, pad = 6 * g_scale;
    fit_text(dl, p.x + 16 * g_scale, cx[0] - fs * 0.3f, p.y, h, P->label, label, false);
    int result = -1;
    bool row_foc = false, row_hov = false;
    for (int column = 0; column < count; column++) {
        int c = controller ? 2 : column;
        ImVec2 a(cx[column], p.y), b(cx[column] + cw, p.y + h);
        const char *t = cells[c] && cells[c][0] ? cells[c] : "-";
        if (!enabled[c]) {   // not a nav item: the cursor skips it
            dl->AddRect(a, b, P->border_idle, r);
            fit_text(dl, a.x + pad, b.x - pad, a.y, h, kGrey, t, true);
            continue;
        }
        ImGui::SetCursorScreenPos(a);
        ImGui::PushID(c);
        ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4(0, 0, 0, 0));   // the cell's own border shows the selection
        bool clicked = ImGui::InvisibleButton("##cell", ImVec2(cw, h), ImGuiButtonFlags_EnableNav);
        ImGui::PopStyleColor();
        char id[80];
        snprintf(id, sizeof id, "%s#%d", key, c);
        bool to_key = g_focus_key[0] && !strcmp(id, g_focus_key);
        if (g_take_first || to_key) {
            ImGui::SetFocusID(ImGui::GetItemID(), ImGui::GetCurrentWindow());
            ImGui::SetNavCursorVisible(true);
            if (g_take_first) ImGui::SetScrollY(0);
            else ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleEdgeY);
            g_take_first = false;
            if (to_key) g_focus_key[0] = 0;
        }
        bool hov = ImGui::IsItemHovered(), foc = ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
        row_foc |= foc;
        row_hov |= hov;
        if (clicked) result = c;
        if (foc && (ImGui::IsKeyPressed(ImGuiKey_Delete, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false) ||
                    ImGui::IsKeyPressed(ImGuiKey_GamepadFaceUp, false)))
            result = 10 + c;
        dl->AddRectFilled(a, b, hov || foc ? P->fill_on : P->fill_idle, r);
        dl->AddRect(a, b, hov || foc ? kGold : P->border_idle, r, 0, hov || foc ? 2.0f : 1.0f);
        fit_text(dl, a.x + pad, b.x - pad, a.y, h, P->cell_text, t, true);
        ImGui::PopID();
    }
    char k[64];
    snprintf(k, sizeof k, "TOTK_%s", key);
    offer_help(ImGui::GetID("##bind_help"), row_hov, row_foc, label, desc, k, nullptr, nullptr, false, note);
    ImGui::SetCursorScreenPos(p);
    ImGui::Dummy(ImVec2(w, h));
    ImGui::PopID();
    return result;
}

// Use the same columns, cap border, text fitting and palette as configurable bindings.
// Only the entire row is a navigation item; no cap is an editable cell.
void opt_shortcut_row(const char *id, UiTextId title, UiTextId desc, const char *key_hint,
                      const char *const *buttons, int count) {
    ImGui::PushID(id);
    ImVec2 p = ImGui::GetCursorScreenPos();
    float fs = ImGui::GetFontSize(), w = ImGui::GetContentRegionAvail().x;
    float h = floorf(fs * 2.65f), cx[2], cw;
    bind_columns(p.x, w, 1, cx, &cw);
    ImGui::InvisibleButton("##shortcut", ImVec2(w, h), ImGuiButtonFlags_EnableNav);
    bool hovered = ImGui::IsItemHovered();
    bool focused = ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
    bool to_key = g_focus_key[0] && !strcmp(id, g_focus_key);
    if (g_take_first || to_key) {
        ImGui::SetFocusID(ImGui::GetItemID(), ImGui::GetCurrentWindow());
        ImGui::SetNavCursorVisible(true);
        if (to_key) {
            ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleEdgeY);
            g_focus_key[0] = 0;
        }
        g_take_first = false;
    }
    offer_help(ImGui::GetItemID(), hovered, focused, menu_text(title), desc, "##shortcut", nullptr, nullptr,
               false, menu_text(UI_MENU_SHORTCUT_READONLY));
    ImDrawList *dl = ImGui::GetWindowDrawList();
    float r = (P->dk ? 6 : 8)*g_scale, pad = 8*g_scale;
    if (hovered || focused) dl->AddRect(p, ImVec2(p.x+w, p.y+h), P->border_idle, r);
    fit_text(dl, p.x+16*g_scale, cx[0]-fs*0.3f, p.y, fs*1.5f, P->label, menu_text(title), false);
    dl->AddText(ImGui::GetFont(), fs*0.75f, ImVec2(p.x+16*g_scale, p.y+fs*1.55f), kDim, key_hint);
    float gap = fs*1.2f, cap_w = (cw-gap*(count-1))/count, cap_h = floorf(fs*1.9f), y = p.y+(h-cap_h)*0.5f;
    for (int i = 0; i<count; i++) {
        float x = cx[0]+i*(cap_w+gap);
        dl->AddRect(ImVec2(x, y), ImVec2(x+cap_w, y+cap_h), P->border_idle, r);
        fit_text(dl, x+pad, x+cap_w-pad, y, cap_h, P->cell_text, buttons[i], true);
        if (i+1<count) fit_text(dl, x+cap_w, x+cap_w+gap, y, cap_h, kDim, "+", true);
    }
    ImGui::PopID();
}
