"""Presentation-only charts; all values come from existing fitted artifacts."""
import plotly.graph_objects as go

TEAL = "#087f83"
INK = "#203d56"
ORANGE = "#cc7043"
MODEL_COLORS = ["#a5b2c0", "#7891a8", TEAL, "#7882ae"]


def style_chart(fig, height=320, y_title=None):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif",color="#60768a",size=11),
        margin=dict(l=12,r=16,t=12,b=28), hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        hoverlabel=dict(bgcolor="#10283e",font_color="white"))
    fig.update_xaxes(showgrid=False, zeroline=False, title_font_size=10, tickfont_size=10)
    fig.update_yaxes(gridcolor="#edf1f5", zeroline=False, title_text=y_title, title_font_size=10, tickfont_size=10)
    return fig


def rul_chart(history, cap, truth=False):
    fig = go.Figure()
    fig.add_scatter(x=history.cycle, y=history.interval_upper, mode="lines", line_width=0, showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=history.cycle, y=history.interval_lower, mode="lines", line_width=0,
        fill="tonexty", fillcolor="rgba(8,127,131,.13)", name="Heuristic spread", hoverinfo="skip")
    fig.add_scatter(x=history.cycle, y=history.ensemble_mean, mode="lines", line=dict(color=TEAL,width=2.5),
        name="Ensemble RUL", hovertemplate="%{y:.1f} cycles<extra>Ensemble</extra>")
    fig.add_scatter(x=history.cycle.iloc[-1:], y=history.ensemble_mean.iloc[-1:], mode="markers",
        marker=dict(size=7,color=TEAL,line=dict(color="white",width=2)),showlegend=False,hoverinfo="skip")
    if truth:
        fig.add_scatter(x=history.cycle,y=history.rul_raw,mode="lines",name="True RUL · evaluation only",
            line=dict(color=ORANGE,width=1.5,dash="dash"),hovertemplate="%{y:.1f} cycles<extra>True RUL</extra>")
    style_chart(fig,320,"Remaining cycles")
    fig.update_xaxes(title_text="Observed cycle")
    fig.update_yaxes(range=[0,max(cap*1.08,history.rul_raw.max()*1.05 if truth else 0)])
    return fig


def health_chart(prefix, result):
    fig = go.Figure()
    fig.add_scatter(x=prefix.cycle,y=result["health_history"],mode="lines",line=dict(color=TEAL,width=2),
        name="PCA health",fill="tozeroy",fillcolor="rgba(8,127,131,.05)",hovertemplate="%{y:.1f} / 100<extra>Health</extra>")
    flagged = result["anomaly_history"] < 0
    fig.add_scatter(x=prefix.cycle.to_numpy()[flagged],y=result["health_history"][flagged],mode="markers",
        name="Isolation Forest flag",marker=dict(color=ORANGE,size=5))
    style_chart(fig,240,"Relative health")
    fig.update_yaxes(range=[-3,103]); fig.update_xaxes(title_text="Observed cycle")
    return fig


def model_chart(result, cap):
    models = list(result["predictions"])
    fig = go.Figure(go.Bar(x=list(result["predictions"].values()),y=models,orientation="h",
        marker_color=MODEL_COLORS,text=[f"{v:.1f}" for v in result["predictions"].values()],
        textposition="outside",textfont=dict(color=INK,size=12),cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f} cycles<extra></extra>",width=.45))
    style_chart(fig,260," ")
    fig.update_layout(hovermode="closest",margin=dict(l=5,r=30,t=15,b=25))
    fig.update_yaxes(autorange="reversed",showgrid=False,title_text=None)
    fig.update_xaxes(range=[0,cap*1.12],title_text="Predicted remaining cycles",showgrid=True,gridcolor="#edf1f5")
    return fig


def sensor_chart(cycles, raw, smooth, unit):
    fig = go.Figure()
    fig.add_scatter(x=cycles,y=raw,name="Observed",mode="lines",line=dict(color="#becad4",width=1),opacity=.8)
    fig.add_scatter(x=cycles,y=smooth,name="Causal rolling mean",mode="lines",line=dict(color=TEAL,width=2))
    style_chart(fig,255,unit)
    fig.update_xaxes(title_text="Observed cycle")
    return fig


def shap_chart(contributions):
    top = contributions.nlargest(10,"Magnitude").sort_values("Contribution")
    fig = go.Figure(go.Bar(x=top.Contribution,y=top.Sensor,orientation="h",
        marker_color=[TEAL if x >= 0 else ORANGE for x in top.Contribution],
        text=[f"{x:+.2f}" for x in top.Contribution],textposition="outside",cliponaxis=False,
        hovertemplate="%{y}: %{x:+.3f} cycles<extra></extra>"))
    style_chart(fig,390)
    fig.update_layout(hovermode="closest",margin=dict(l=10,r=40,t=10,b=30))
    fig.update_xaxes(title_text="Contribution to predicted RUL (cycles)",showgrid=True,gridcolor="#edf1f5",zeroline=True,zerolinecolor="#9dabb7")
    fig.update_yaxes(showgrid=False)
    return fig
