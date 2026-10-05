import streamlit as st
from ui.i18n import dark_mode
from src.embedding_view import pca_2d, projection_rows, selected_map_word
from src.tables import rows_csv

@st.cache_data(show_spinner=False, max_entries=8)
def project_vectors(vectors):
    return pca_2d(vectors)

def embedding_map(model, query, key):
    """The query word and its nearest neighbours on a 2D PCA map of the trained vectors."""
    st.markdown("#### Word vector map · PCA 2D")
    if query is None:
        st.info("Enter a word from the trained vocabulary to see it and its nearest words on the map.")
        return
    coordinates, ratios = project_vectors(model.vectors)
    left, right = st.columns([2, 1])
    maximum = min(300, len(model.words))
    limit = int(left.number_input("Words on the map (query + nearest)", min_value=2, max_value=maximum,
                                  value=min(40, maximum), key=f"map_count_{key[:12]}"))
    labels = right.checkbox("Show word labels", value=True, key=f"map_labels_{key[:12]}")
    rows = projection_rows(model, coordinates, query=query, limit=limit, mode="neighbors")
    st.caption(f"The query and its {limit - 1} nearest words by cosine. PCA fitted once to all {len(model.words):,} L2-normalized "
               f"word vectors: PC1 explains {ratios[0]:.1%}, PC2 {ratios[1]:.1%}, total {sum(ratios):.1%}.")
    if sum(ratios) == 0:
        st.info("These vectors have no variance after normalization; all points overlap at the origin.")
    spec = {
        "data": {"values": rows}, "height": 480,
        "encoding": {
            "x": {"field": "pc1", "type": "quantitative", "title": f"PC1 ({ratios[0]:.1%} variance)", "scale": {"zero": False}},
            "y": {"field": "pc2", "type": "quantitative", "title": f"PC2 ({ratios[1]:.1%} variance)", "scale": {"zero": False}},
        },
        "layer": [{
            "mark": {"type": "circle", "opacity": 0.85},
            "params": [{"name": "map_zoom", "select": "interval", "bind": "scales"},
                       {"name": "pick_word", "select": {"type": "point", "fields": ["word"], "on": "click", "toggle": False}}],
            "encoding": {
                "color": {"field": "group", "type": "nominal", "title": None,
                          "scale": {"domain": ["Query", "Neighbor"], "range": ["#e37942", "#218c86"]}},
                "size": {"condition": {"test": "datum.group === 'Query'", "value": 220}, "value": 85},
                "tooltip": [{"field": "word", "title": "Word"}, {"field": "corpus_count", "title": "Corpus frequency"},
                            {"field": "cosine_similarity", "type": "quantitative", "format": ".4f", "title": "Cosine to query (original vectors)"},
                            {"field": "pc1", "format": ".4f"}, {"field": "pc2", "format": ".4f"}],
            },
        }],
    }
    ink = '#c9d8dc' if dark_mode() else '#304651'
    spec['background'] = 'transparent'
    spec['config'] = {'axis': {'labelColor': ink, 'titleColor': ink, 'gridColor': '#566b7630'},
                      'legend': {'labelColor': ink, 'titleColor': ink}, 'view': {'stroke': None}}
    if labels:
        spec["layer"].append({"mark": {"type": "text", "align": "left", "dx": 8, "dy": -7, "fontSize": 11, "color": ink},
                              "encoding": {"text": {"field": "word"}}})
    event = st.vega_lite_chart(spec=spec, width='stretch', on_select="rerun",
                               selection_mode="pick_word", key=f"word_map_{key[:12]}")
    st.caption("Orange: the query; green: its nearest words. Hover for frequency and cosine; scroll to zoom, drag to pan, "
               "double-click to reset; click a point to list its own nearest words. Distances in 2D only approximate the "
               "64-dimensional vectors; the table on the left is ranked in the original vector space.")
    word = selected_map_word(event.selection.get("pick_word", {}))
    if word in {row["word"] for row in rows}:
        st.write(f"Selected word: **{word}**")
        st.dataframe(model.most_similar(word, 10), hide_index=True, width='stretch')
    st.download_button("Download PCA coordinates (CSV)", rows_csv(rows), "word2vec_pca_2d.csv", "text/csv")
