import figures
import ss_1_ensemble
import ss_2_aggregation
import ss_3_scores

if __name__ == "__main__":
    # 1. Run ensemble prediction
    ss_1_ensemble.main()

    # 2. Run aggregation
    ss_2_aggregation.main()

    # 3. Score results
    ss_3_scores.main()

    # 4. Plot results
    figures.plot_panel_model()
    figures.plot_panel_boxplot()
    figures.plot_pit_ens()
    figures.plot_ensemble_members()
