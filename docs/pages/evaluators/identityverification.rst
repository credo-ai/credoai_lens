
Identity verification
=====================


Pair-wise-comparison-based identity verification evaluator for Credo AI

This evaluator takes in identity verification data and
    provides functionality to perform performance and fairness assessment

Parameters
----------
pairs : pd.DataFrame of shape (n_pairs, 4)
    Dataframe where each row represents a data sample pair and associated subjects
    Type of data sample is decided by the ComparisonModel's `compare` function, which takes
    data sample pairs and returns their similarity scores. Examples are selfies, fingerprint scans,
    or voices of a person.
    Required columns:
        source-subject-id: unique identifier of the source subject
        source-subject-data-sample: data sample from the source subject
        target-subject-id: unique identifier of the target subject
        target-subject-data-sample: data sample from the target subject
subjects_sensitive_features : pd.DataFrame of shape (n_subjects, n_sensitive_feature_names), optional
    Sensitive features of all subjects present in pairs dataframe
    If provided, disaggregated performance assessment is also performed.
    This can be the columns you want to perform segmentation analysis on, or
    a feature related to fairness like 'race' or 'gender'
    Required columns:
        subject-id: id of subjects. Must cover all the subjects inlcluded in `pairs` dataframe
        other columns with arbitrary names for sensitive features
similarity_thresholds : list
    list of similarity score thresholds
    Similarity equal or greater than a similarity score threshold means match
comparison_levels : list
    list of comparison levels. Options:
        sample: it means a match is observed for every sample pair. Sample-level comparison represent
            a use case where only two samples (such as a real time selfie and stored ID image) are
            used to confirm an identity.
        subject: it means if any pairs of samples for the same subject are a match, the subject pair
            is marked as a match. Some identity verification use cases improve overall accuracy by storing
            multiple samples per identity. Subject-level comparison mirrors this behavior.
