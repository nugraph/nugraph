import pandas as pd
import torch
from torch_geometric.data import Batch, HeteroData
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.graph_objects import FigureWidget
import warnings

class GraphPlot:
    def __init__(self,
                 planes: list[str],
                 classes: list[str],
                 filter_threshold: float = 0.5):
        self._planes = planes
        self._classes = classes
        self._labels = pd.CategoricalDtype(['background']+classes, ordered=True)
        self._cmap = { c: px.colors.qualitative.Plotly[i] for i, c in enumerate(classes) }
        self._cmap['background'] = 'lightgrey'
        self.filter_threshold = filter_threshold

        # temporarily silence this pandas warning triggered by plotly,
        # which we don't have any power to fix but will presumably
        # be fixed on their end at some point
        warnings.filterwarnings("ignore", ".*The default of observed=False is deprecated and will be changed to True in a future version of pandas.*")
        self._truth_cols = ( 'g4_id', 'parent_id', 'pdg' )

    def to_dataframe(self, data: HeteroData):
        def to_categorical(arr):
            return pd.Categorical.from_codes(codes=arr+1, dtype=self._labels)
        if isinstance(data, Batch):
            raise Exception('to_dataframe does not support batches!')

        hit = data["hit"].to_dict()
        df = pd.DataFrame(hit["id"], columns=["id"])
        df["plane"] = [self._planes[i] for i in hit["plane"]]
        df[["proj", "drift"]] = hit["pos"]
        if "c" in hit:
            df[["x", "y", "z"]] = hit["c"]
        df["y_filter"] = hit["y_semantic"] != -1
        df['y_semantic'] = to_categorical(hit['y_semantic'])
        df['y_instance'] = data.y_i().numpy().astype(str)

        # add detailed truth information if it's available
        for col in self._truth_cols:
            if col in hit.keys():
                df[col] = hit[col].numpy()

        # add model prediction if it's available
        if 'x_semantic' in hit.keys():
            df['x_semantic'] = to_categorical(hit['x_semantic'].argmax(dim=-1).detach())
            df[self._classes] = hit['x_semantic'].detach()
        if 'x_filter' in hit.keys():
            df['x_filter'] = hit['x_filter'].detach()
        if "ox" in hit.keys():
            df["i"] = data.x_i().numpy().astype(str)

        # add object condensation embedding
        if "ox" in hit.keys():
            coords = data["hit"].ox.cpu()
            pca = PCA(n_components=2)
            df["c1"], df["c2"] = pca.fit_transform(coords).transpose()
            beta = data["hit"].of.cpu()
            df["logbeta"] = beta.log10()

        # add event metadata
        md = data['metadata']
        df['run'] = md.run.item()
        df['subrun'] = md.subrun.item()
        df['event'] = md.event.item()

        return df

    def plot(self,
             data: HeteroData,
             target: str = 'hits',
             how: str = 'none',
             filter: str = 'show',
             xyz: bool = False,
             width: int = None,
             height: int = None,
             title: bool = True) -> FigureWidget:

        df = self.to_dataframe(data)

        # no colour
        if target == 'hits':
            opts = {
                'title': 'Graph hits',
            }

        # semantic labels
        elif target == 'semantic':
            if how == 'true':
                opts = {
                    'title': 'True semantic labels',
                    'labels': { 'y_semantic': 'Semantic label' },
                    'color': 'y_semantic',
                    'color_discrete_map': self._cmap,
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted semantic labels',
                    'labels': { 'x_semantic': 'Semantic label' },
                    'color': 'x_semantic',
                    'color_discrete_map': self._cmap,
                }
            elif how in self._classes:
                opts = {
                    'title': f'Predicted semantic label strength for {how} class',
                    'labels': { how: f'{how} probability' },
                    'color': how,
                    'color_continuous_scale': px.colors.sequential.Reds,
                }
            else:
                raise Exception('for semantic labels, "how" must be one of "true", "pred" or the name of a class.')

        # instance labels
        elif target == 'instance':
            if how == 'true':
                opts = {
                    'title': 'True instance labels',
                    'labels': { 'y_instance': 'Instance label' },
                    'color': 'y_instance',
                    'symbol': 'y_semantic',
                    'color_discrete_map': self._cmap,
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted instance labels',
                    'labels': { 'i': 'Instance label' },
                    'color': 'i',
                    'color_discrete_map': self._cmap,
                }
            elif how == 'beta':
                opts = {
                    'title': 'Object condensation beta values',
                    'color': 'logbeta',
                    'color_continuous_scale': px.colors.sequential.Reds,
                }
            elif how == 'pca':
                opts = {
                    'title': 'Object condensation coordinates',
                    'labels': { 'y_instance': 'Instance label' },
                    'color': 'y_instance',
                    'color_discrete_map': self._cmap,
                }
            else:
                raise Exception('for instance labels, "how" must be one of "true", "pred", "beta" or "pca".')

        # filter labels
        elif target == 'filter':
            if how == 'true':
                opts = {
                    'title': 'True filter labels',
                    'labels': { 'y_filter': 'Filter label' },
                    'color': 'y_filter',
                    'color_discrete_map': { 0: 'coral', 1: 'mediumseagreen' },
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted filter labels',
                    'labels': { 'x_filter': 'Filter label' },
                    'color': 'x_filter',
                    'color_continuous_scale': px.colors.sequential.Reds,
                }
            else:
                raise Exception('for filter labels, "how" must be one of "true" or "pred".')

        else:
            raise Exception('"target" must be one of "hits", "semantic", "instance" or "filter".')

        if filter == 'none':
            # don't do any filtering
            pass
        elif filter == 'show':
            # show hits predicted to be background in grey
            if target == 'semantic' and how == 'pred':
                df.loc[df.x_filter < self.filter_threshold, "x_semantic"] = 'background'
        elif filter == 'true':
            # remove true background hits
            df = df[df.y_filter.values]
            opts['title'] += ' (filtered by truth)'
        elif filter == 'pred':
            # remove predicted background hits
            df = df[df.x_filter > self.filter_threshold]
            opts['title'] += ' (filtered by prediction)'
        else:
            raise Exception('"filter" must be one of "none", "show", "true" or "pred".')

        if xyz:
            opts["x"] = "x"
            opts["y"] = "y"
            opts["z"] = "z"
        elif how == "pca":
            opts["x"] = "c1"
            opts["y"] = "c2"
        else:
            opts["x"] = "proj"
            opts["y"] = "drift"
            opts["facet_col"] = "plane"

        if not title:
            opts.pop('title')

        # set hover data
        opts['hover_data'] = {
            'y_semantic': True,
            "y_instance": True,
            'proj': ':.1f',
            'drift': ':.1f',
        }
        opts['labels'] = {
            'y_filter': 'filter truth',
            'y_semantic': 'semantic truth',
            'y_instance': 'instance truth',
        }
        if 'x_filter' in df:
            opts['hover_data']['x_filter'] = True
            opts['labels']['x_filter'] = 'filter prediction'
        if 'x_semantic' in df:
            opts['hover_data']['x_semantic'] = True
            opts['labels']['x_semantic'] = 'semantic prediction'
        if 'i' in df:
            opts['hover_data']['i'] = ':.4f'
            opts['labels']['i'] = 'instance prediction'
        for col in self._truth_cols:
            if col in df:
                opts['hover_data'][col] = True

        if xyz:
            fig = px.scatter_3d(df, width=width, height=height, **opts)
            fig.update_traces(marker_size=1)
        else:
            fig = px.scatter(df, width=width, height=height, **opts)
            fig.update_xaxes(matches=None)
            for a in fig.layout.annotations:
                a.text = a.text.replace('plane=', '')

        # set the legend to horizontal
        fig.update_layout(
            legend_orientation='h',
            legend_yanchor='bottom', legend_y=1.05,
            legend_xanchor='right', legend_x=1,
            margin_l=20, margin_r=20, margin_t=20, margin_b=20,
            title_automargin=title,
        )

        return FigureWidget(fig)