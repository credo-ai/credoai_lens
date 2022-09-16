from fairlearn.metrics import MetricFrame


def _create_metric_frame(metrics, y_pred, y_true, sensitive_features):
    """Creates metric frame from dictionary of key:Metric"""
    metrics = {name: metric.fun for name, metric in metrics.items()}
    return MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )


def _setup_metric_frames(self):
    self.metric_frames = {}
    if self.y_pred is not None and self.performance_metrics:
        self.metric_frames["pred"] = _create_metric_frame(
            self.performance_metrics,
            self.y_pred,
            self.y_true,
            sensitive_features=self.sensitive_features,
        )

        # for metrics that require the probabilities
        self.prob_metric_frame = None
        if self.y_prob is not None and self.prob_metrics:
            self.metric_frames["prob"] = self._create_metric_frame(
                self.prob_metrics,
                self.y_prob,
                sensitive_features=self.sensitive_features,
            )
