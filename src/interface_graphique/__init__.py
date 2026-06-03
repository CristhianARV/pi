import gradio as gr
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rag import RAGPipeline, SEARCH_MODES


CONV_ROOT = Path(__file__).parent.parent.parent / "data" / "conversations"
DEFAULT_CATEGORY = "General"

MODE_LABELS = {
    "semantic": "Recherche intelligente",
    "text": "Recherche exacte",
    "hybrid": "Recherche avancée",
}

MODE_DESCRIPTIONS = {
    "semantic": (
        "Recherche intelligente : l’assistant cherche les passages qui ont le même sens que votre question. "
        "Ce mode est adapté si vous décrivez un problème avec vos propres mots, sans connaître le terme exact."
    ),
    "text": (
        "Recherche exacte : l’assistant cherche les mots présents dans votre question directement dans les documents. "
        "Ce mode est adapté pour retrouver une référence précise, un code erreur, un nom de pièce, "
        "un modèle machine ou un terme technique exact."
    ),
    "hybrid": (
        "Recherche avancée : l’assistant combine la recherche intelligente et la recherche exacte. "
        "Il effectue une recherche par le sens, puis une recherche par mots-clés, avant de produire une réponse finale. "
        "Ce mode est souvent le plus complet, mais il peut être plus lent."
    ),
}


def _mode_choices() -> list[tuple[str, str]]:
    return [
        (MODE_LABELS.get(mode, mode), mode)
        for mode in SEARCH_MODES
    ]


def _mode_description(mode: str) -> str:
    return MODE_DESCRIPTIONS.get(
        mode,
        "Choisissez un mode de recherche."
    )

def _dyslexic_css(enabled: bool) -> str:
    if not enabled:
        return "<style></style>"

    return """
    <style>
    body,
    .gradio-container,
    .gradio-container textarea,
    .gradio-container input,
    .gradio-container select,
    .gradio-container button,
    .gradio-container label,
    .gradio-container p,
    .gradio-container span {
        font-family: Verdana, Arial, sans-serif !important;
        letter-spacing: 0.045em !important;
        word-spacing: 0.08em !important;
        line-height: 1.65 !important;
    }

    .gradio-container textarea,
    .gradio-container input,
    .gradio-container .prose,
    .gradio-container .message,
    .gradio-container .markdown {
        font-size: 1.04rem !important;
    }
    </style>
    """
# ============ Persistance disque ============

def _load_all() -> dict[str, list[dict]]:
    CONV_ROOT.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}
    for cat_dir in sorted(CONV_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        convs = []
        for f in cat_dir.glob("*.json"):
            try:
                with open(f, encoding="utf-8") as fp:
                    convs.append(json.load(fp))
            except (json.JSONDecodeError, OSError):
                continue
        convs.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        out[cat_dir.name] = convs
    if not out:
        out[DEFAULT_CATEGORY] = []
    return out


def _save_conv(conv: dict, category: str) -> None:
    folder = CONV_ROOT / category
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{conv['id']}.json", "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)


def _delete_conv_file(conv_id: str, category: str) -> None:
    f = CONV_ROOT / category / f"{conv_id}.json"
    if f.exists():
        f.unlink()


def _new_conv() -> dict:
    return {
        "id": uuid4().hex[:8],
        "title": "New conversation",
        "messages": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _conv_choices(convs: list[dict]) -> list[tuple[str, str]]:
    return [(c["title"], c["id"]) for c in convs]


def _find_conv(all_convs: dict, conv_id: str) -> tuple[str | None, dict | None]:
    for cat, convs in all_convs.items():
        for c in convs:
            if c["id"] == conv_id:
                return cat, c
    return None, None


# ============ UI ============

def _build_blocks(pipeline: RAGPipeline) -> gr.Blocks:
    initial = _load_all()
    initial_cat = next(iter(initial))
    initial_convs = initial[initial_cat]

    if not initial_convs:
        first = _new_conv()
        _save_conv(first, initial_cat)
        initial[initial_cat] = [first]
        initial_convs = [first]

    initial_active = initial_convs[0]["id"]

    with gr.Blocks(title="RAG Assistant", fill_height=True) as app:
        dyslexic_style = gr.HTML(value=_dyslexic_css(False))
        all_state = gr.State(initial)
        cat_state = gr.State(initial_cat)
        active_state = gr.State(initial_active)

        with gr.Row():
            # --- Sidebar ---
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Catégories")
                category_dd = gr.Dropdown(
                    choices=list(initial.keys()),
                    value=initial_cat,
                    label="Niveau",
                    allow_custom_value=True,
                    container=False,
                )
                with gr.Row():
                    add_cat_btn = gr.Button("➕ Catégorie", size="sm")
                    del_cat_btn = gr.Button("🗑️", size="sm")
                

                gr.Markdown("### Discussions")
                new_btn = gr.Button("➕ Nouvelle discussion", variant="primary")
                conv_list = gr.Radio(
                    choices=_conv_choices(initial_convs),
                    value=initial_active,
                    show_label=False,
                    container=False,
                )
                del_conv_btn = gr.Button("🗑️ Supprimer", size="sm")
                gr.Markdown("### Accessibilité")
                dyslexic_font = gr.Checkbox(
                    value=False,
                    label="Police adaptée dyslexie",
                    info="Augmente l’espacement, la taille du texte et utilise une police plus lisible.",
                )
                gr.Markdown(
                    """
                    **Mode d’affichage**

                    Dans Paramètres ⚙️ :

                    - `light` pour le mode jour
                    - `dark` pour le mode nuit
                    """
                )

            # --- Main chat ---
            with gr.Column(scale=4):
                gr.Markdown("## RAG Assistant")
                chatbot = gr.Chatbot(
                    height=520,
                    value=initial_convs[0]["messages"] if initial_convs else [],
                )
                msg = gr.Textbox(
                    placeholder="Posez votre question…",
                    show_label=False,
                    autofocus=True,
                    submit_btn=True,
                )
                with gr.Row():
                    search_mode = gr.Dropdown(
                        choices=_mode_choices(),
                        value="semantic",
                        label="Mode de recherche",
                    )
                    use_rerank = gr.Checkbox(
                        value=pipeline.use_reranker,
                        label="Reranker",
                        info="Plus pertinent, plus lent",
                    )
                    collection = gr.Dropdown(
                        choices=pipeline.available_collections(),
                        value=pipeline.collection_name,
                        label="Collection",
                    )
                mode_description = gr.Markdown(
                    value=_mode_description("semantic"),
                    elem_classes=["mode-description"],
                )

        # ============ Handlers ============
        def on_search_mode_change(mode):
            return _mode_description(mode)

        def on_dyslexic_change(enabled):
            return _dyslexic_css(enabled)

        def on_category_change(cat, all_convs):
            if cat not in all_convs:
                all_convs[cat] = []
                (CONV_ROOT / cat).mkdir(parents=True, exist_ok=True)
            convs = all_convs[cat]
            if convs:
                active = convs[0]["id"]
                messages = convs[0]["messages"]
            else:
                fresh = _new_conv()
                _save_conv(fresh, cat)
                all_convs[cat] = [fresh]
                convs = [fresh]
                active = fresh["id"]
                messages = []
            return (
                all_convs, cat, active,
                gr.update(choices=list(all_convs.keys()), value=cat),
                gr.update(choices=_conv_choices(convs), value=active),
                messages,
            )

        def on_add_category(all_convs):
            base = "Nouvelle catégorie"
            name = base
            i = 1
            while name in all_convs:
                i += 1
                name = f"{base} {i}"
            all_convs[name] = []
            (CONV_ROOT / name).mkdir(parents=True, exist_ok=True)
            return on_category_change(name, all_convs)

        def on_delete_category(cat, all_convs):
            cat_path = CONV_ROOT / cat
            if cat_path.exists():
                for f in cat_path.glob("*.json"):
                    f.unlink()
                cat_path.rmdir()
            all_convs.pop(cat, None)
            if not all_convs:
                all_convs[DEFAULT_CATEGORY] = []
                (CONV_ROOT / DEFAULT_CATEGORY).mkdir(parents=True, exist_ok=True)
            new_cat = next(iter(all_convs))
            return on_category_change(new_cat, all_convs)

        def on_new_conv(cat, all_convs):
            c = _new_conv()
            _save_conv(c, cat)
            all_convs[cat].insert(0, c)
            return (
                all_convs, c["id"],
                gr.update(choices=_conv_choices(all_convs[cat]), value=c["id"]),
                [],
            )

        def on_select_conv(conv_id, cat, all_convs):
            if conv_id is None:
                return gr.update(), gr.update()
            for c in all_convs.get(cat, []):
                if c["id"] == conv_id:
                    return conv_id, c["messages"]
            return gr.update(), gr.update()

        def on_delete_conv(cat, active, all_convs):
            _delete_conv_file(active, cat)
            all_convs[cat] = [c for c in all_convs[cat] if c["id"] != active]
            if not all_convs[cat]:
                fresh = _new_conv()
                _save_conv(fresh, cat)
                all_convs[cat] = [fresh]
            new_active = all_convs[cat][0]["id"]
            return (
                all_convs, new_active,
                gr.update(choices=_conv_choices(all_convs[cat]), value=new_active),
                all_convs[cat][0]["messages"],
            )

        def on_submit(user_msg, history, mode, use_rerank_val, coll,
              all_convs, cat, active):
            if not user_msg.strip():
                yield history, all_convs, gr.update()
                return

            pipeline.set_collection(coll)
            pipeline.set_search_mode(mode)
            pipeline.set_use_reranker(use_rerank_val)

            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ""},
            ]
            yield history, all_convs, gr.update()

            for content in pipeline.stream(user_msg):
                history[-1]["content"] = content
                yield history, all_convs, gr.update()

            _, conv = _find_conv(all_convs, active)
            if conv is not None:
                conv["messages"] = history
                if conv["title"] == "New conversation":
                    conv["title"] = user_msg[:40] + ("…" if len(user_msg) > 40 else "")
                _save_conv(conv, cat)

            yield history, all_convs, gr.update(
                choices=_conv_choices(all_convs[cat]), value=active
            )

        # ============ Wiring ============

        category_dd.change(
            on_category_change,
            inputs=[category_dd, all_state],
            outputs=[all_state, cat_state, active_state, category_dd, conv_list, chatbot],
        )

        add_cat_btn.click(
            on_add_category,
            inputs=[all_state],
            outputs=[all_state, cat_state, active_state, category_dd, conv_list, chatbot],
        )

        del_cat_btn.click(
            on_delete_category,
            inputs=[cat_state, all_state],
            outputs=[all_state, cat_state, active_state, category_dd, conv_list, chatbot],
        )

        new_btn.click(
            on_new_conv,
            inputs=[cat_state, all_state],
            outputs=[all_state, active_state, conv_list, chatbot],
        )

        conv_list.change(
            on_select_conv,
            inputs=[conv_list, cat_state, all_state],
            outputs=[active_state, chatbot],
        )

        del_conv_btn.click(
            on_delete_conv,
            inputs=[cat_state, active_state, all_state],
            outputs=[all_state, active_state, conv_list, chatbot],
        )

        search_mode.change(
            on_search_mode_change,
            inputs=[search_mode],
            outputs=[mode_description],
        )

        dyslexic_font.change(
            on_dyslexic_change,
            inputs=[dyslexic_font],
            outputs=[dyslexic_style],
        )

        

        msg.submit(
            on_submit,
            inputs=[msg, chatbot, search_mode, use_rerank, collection,
                    all_state, cat_state, active_state],
            outputs=[chatbot, all_state, conv_list],
        ).then(lambda: "", outputs=msg)

    return app


class ChatApp:
    def __init__(self, pipeline: RAGPipeline):
        self._app = _build_blocks(pipeline)

    def mainloop(self, **kwargs):
        self._app.launch(theme=gr.themes.Soft(), **kwargs)