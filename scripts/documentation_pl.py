"""Polish text for the descriptions the thesis reads from the code.

The code documents itself in English and the thesis is written in Polish. Every
test module and case, command, route and setting the thesis lists has its
Polish description here, keyed by what identifies it in the code. Each entry
also carries the first eight hex digits of the SHA-1 of the English text it was
translated from, so that the build can say which translations the code has
outgrown.

Flag descriptions are keyed by the English help text itself: most flags share
one of a few texts, and a changed text simply finds no translation.
"""

from __future__ import annotations

import hashlib

# Collected while the document is built and reported once at the end of it.
_problems: list[str] = []


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def polish(table: dict[str, tuple[str, str]], key: str, english: str) -> str:
    """The Polish text of one description, or the English one, noted, if there is none."""
    entry = table.get(key)
    if entry is None:
        _problems.append(f"no Polish text for {key}")
        return english
    digest, text = entry
    if digest != _digest(english):
        _problems.append(f"the English text of {key} changed (now {_digest(english)})")
    return text


def polish_help(english: str) -> str:
    """The Polish description of a command-line flag."""
    if english in OPTIONS:
        return OPTIONS[english]
    _problems.append(f"no Polish text for the flag help {english!r}")
    return english


def report() -> None:
    """Print the descriptions the build had to leave in English or found out of date."""
    if not _problems:
        return
    found = sorted(set(_problems))
    print(f"WARNING: {len(found)} Polish descriptions are missing or out of date:")
    for problem in found[:20]:
        print(f"  {problem}")
    _problems.clear()


# Test modules, by file name.
MODULES: dict[str, tuple[str, str]] = {
    "test_aggregate.py": ("d5076ea4", "Testy agregacji świec."),
    "test_api_command_root.py": (
        "8a89ff15",
        "Każde zadanie z kolejki pracuje na jeziorze danych, z którym uruchomiono serwer.",
    ),
    "test_api_path_safety.py": (
        "214b452a",
        "Żadne żądanie nie może wskazać ścieżki poza jeziorem danych, a żadna inna strona nie "
        "może takiego żądania wysłać.",
    ),
    "test_api_performance.py": (
        "cd4948f1",
        "Ograniczenia rozmiaru odpowiedzi i zużycia pamięci w interfejsie HTTP.",
    ),
    "test_api_pool_command.py": (
        "8ef21199",
        "Interfejs HTTP nie może wysłać flagi, której wiersz poleceń nie zna, ani pominąć "
        "flagi wybranej przez wywołującego.",
    ),
    "test_api_server.py": ("091996a8", "Testy interfejsu HTTP: budowanie poleceń i trasy danych."),
    "test_backtest_execution.py": ("a112a850", "Gwarancje modelu wykonania w symulacji handlowej."),
    "test_backtest_nulls.py": (
        "adb4a9a8",
        "Kupno i trzymanie jest najsłabszym z czterech punktów odniesienia, a nie jedynym.",
    ),
    "test_backtest_parameters.py": (
        "4b022722",
        "Założenia wykonawcze są parametrami symulacji, a punkty odniesienia nie.",
    ),
    "test_calibration.py": (
        "2a8d8174",
        "Kalibracja dopasowywana jest wewnątrz okna treningowego, nigdy po nim.",
    ),
    "test_calibration_record.py": (
        "0e943b4c",
        "Każda skalibrowana prognoza zapisuje, co zrobiła z nią kalibracja.",
    ),
    "test_cli_api_contract.py": (
        "6f7bab08",
        "Każda flaga wysyłana przez interfejs HTTP musi istnieć w poleceniu wiersza poleceń, "
        "do którego trafia.",
    ),
    "test_cli_exit_codes.py": ("fe0048d4", "Kody wyjścia poleceń potoku w wierszu poleceń."),
    "test_cost_threshold.py": (
        "3baa54a2",
        "Sygnał musi pokryć koszt pełnej transakcji, zanim planer na niego zareaguje.",
    ),
    "test_data_manager.py": ("0a786c30", "Testy inwentarza jeziora danych (DataManager)."),
    "test_depth_scheme.py": (
        "479b4f06",
        "Schemat kubełków głębokości ma jedną definicję, a artefakty zapisują własny.",
    ),
    "test_download_verification.py": (
        "5507717d",
        "Do jeziora danych trafia tylko archiwum, którego suma kontrolna została odczytana i "
        "się zgadza.",
    ),
    "test_early_stopping.py": (
        "01bccbc8",
        "Uczenie zatrzymuje się, gdy strata na danych odłożonych przestaje maleć.",
    ),
    "test_eval_stability.py": (
        "5d0077d8",
        "Jakość prognozy w czasie oraz punktacja wspólna z resztą oceny.",
    ),
    "test_features_build.py": ("69e2a650", "Testy wyznaczania cech."),
    "test_forward_labels.py": (
        "a5231d0c",
        "Jedna definicja etykiety, z której korzysta każdy model i plik etykiet.",
    ),
    "test_hash_utils.py": ("50e48ad9", "Testy funkcji sum kontrolnych."),
    "test_history_shorter_than_range.py": (
        "0eacf26b",
        "Symbol, którego historia zaczyna się wewnątrz żądanego zakresu, nie może przerwać "
        "przebiegu.",
    ),
    "test_holdout_purge.py": (
        "531b884d",
        "Wycinek kalibracyjny musi leżeć poza próbą modelu, który kalibruje.",
    ),
    "test_horizon_set.py": (
        "a9cd4b90",
        "Zbiór horyzontów: pełne tygodnie, wzajemnie współmierne, nazwane od rozpiętości.",
    ),
    "test_integration.py": (
        "5cf329fe",
        "Testy integracyjne bez rzeczywistej sieci, z atrapami albo w trybie próbnym.",
    ),
    "test_inventory_cache.py": (
        "66826c6a",
        "Pamięć podręczna inwentarza musi zauważyć partycję, która zniknęła.",
    ),
    "test_labels.py": ("b15f7265", "Testy budowania etykiet."),
    "test_missing_features.py": (
        "1fd8cdc6",
        "Obie sieci sekwencyjne zamieniają brakującą cechę na tę samą liczbę.",
    ),
    "test_one_grader.py": ("bc633461", "O werdykcie decyduje jeden obiekt i nic poza nim."),
    "test_order_flow_features.py": (
        "7da356a0",
        "Binance udostępnia dla każdej świecy więcej niż OHLCV, a potok to zachowuje.",
    ),
    "test_planner_windows.py": ("c25bf4a5", "Testy funkcji planera okien działania."),
    "test_pooled_training.py": (
        "72ec5c1e",
        "Łączenie symboli ma dodawać obserwacje, nie dodając przecieku z przyszłości.",
    ),
    "test_predict_interval_integration.py": (
        "01007f08",
        "Testy integracyjne: ścieżka polecenia prognozy, wersjonowany zapis i trening "
        "horyzontów.",
    ),
    "test_predict_leakage.py": (
        "8d9d0b42",
        "Testy ograniczenia okna treningowego w pętli kroczącej, wspólnego dla wszystkich "
        "modeli.",
    ),
    "test_prediction_evaluation.py": ("96f8995c", "Testy oceny jakości wersjonowanych prognoz."),
    "test_predictor_training_edges.py": (
        "03c4c2bb",
        "Przypadki brzegowe, w których model mógłby po cichu przestać się uczyć.",
    ),
    "test_qa_checks.py": ("5830058b", "Testy kontroli jakości danych."),
    "test_risk_gates.py": (
        "3d4ac92e",
        "Bramka ryzyka jest względna wobec tego, jak dany rynek wyglądał dotychczas.",
    ),
    "test_risk_is_directional.py": (
        "08250bfe",
        "Ryzyko to szansa ruchu przeciw pozycji, a nie dowolnego dużego ruchu.",
    ),
    "test_row_order.py": (
        "8c9c2f0e",
        "Usuwanie duplikatów nie może zmieniać kolejności wierszy odczytywanych według "
        "pozycji.",
    ),
    "test_run_attribution.py": (
        "4e62fb7f",
        "Każdy artefakt pochodny musi wskazywać przebieg prognozy, z którego powstał.",
    ),
    "test_run_provenance.py": ("48189caa", "Przebieg musi mówić, co go wytworzyło."),
    "test_scale_free_features.py": (
        "64d80ad7",
        "Model może uczyć się wyłącznie na cechach, które na każdym symbolu znaczą to samo.",
    ),
    "test_significance.py": (
        "de5ad201",
        "Przedział ufności, który pomija nakładanie się etykiet, jest gorszy niż żaden.",
    ),
    "test_task_queue.py": ("4e3a3fb7", "Kolejka zadań żyje na serwerze, a nie w przeglądarce."),
    "test_tft_batched_inference.py": (
        "7d8c3c2a",
        "Prognoza TFT liczona jest wsadowo dla całego miesiąca douczania, a nie budowana od "
        "nowa dla każdej świecy.",
    ),
    "test_tft_class_order.py": (
        "3d2a5ed0",
        "Kolumny wyjściowe TFT nazywa koder etykiet, nigdy ich pozycja.",
    ),
    "test_time_utils.py": ("567b80e9", "Testy funkcji obsługi czasu."),
    "test_train_window.py": (
        "718c0205",
        "Skąd bierze się okno treningowe, które nie jest już stałą.",
    ),
    "test_unlabelled_bars.py": (
        "e4d1f79e",
        "Świeca bez etykiety nadal należy do szeregu (SPECIFICATION.md, punkt 2.2).",
    ),
    "test_versioned_predictions.py": (
        "e14445cf",
        "Testy funkcji wersjonowanych katalogów prognoz i tras modeli w interfejsie HTTP.",
    ),
    "test_walk_forward.py": ("d03b425d", "Pętla krocząca, testowana raz zamiast pięciu razy."),
    "test_window_seed.py": (
        "de2d2906",
        "Losowość każdego trenowanego modelu należy wyłącznie do niego.",
    ),
}

# Test cases, as module::name without the test_ prefix.
CASES: dict[str, tuple[str, str]] = {
    "test_aggregate.py::aggregate_1m_to_1d": ("da39a3ee", "Agregacja świec 1m do interwału 1d."),
    "test_aggregate.py::aggregate_1m_to_1h": ("da39a3ee", "Agregacja świec 1m do interwału 1h."),
    "test_aggregate.py::aggregate_1m_to_1w": ("da39a3ee", "Agregacja świec 1m do interwału 1w."),
    "test_aggregate.py::aggregate_empty_dataframe": ("da39a3ee", "Agregacja pustej ramki danych."),
    "test_aggregate.py::aggregate_pipeline_writes_requested_intervals": (
        "da39a3ee",
        "Potok agregacji zapisuje wszystkie żądane interwały.",
    ),
    "test_aggregate.py::normalize_target_intervals_accepts_cli_shapes": (
        "da39a3ee",
        "Lista interwałów docelowych przyjmuje każdą postać zapisu dopuszczaną przez wiersz "
        "poleceń.",
    ),
    "test_api_command_root.py::every_command_carries_the_configured_root": (
        "da39a3ee",
        "Każde polecenie z kolejki dostaje skonfigurowany katalog główny danych.",
    ),
    "test_api_command_root.py::the_queued_jobs_and_the_endpoints_read_the_same_lake": (
        "ed03041c",
        "Zadania z kolejki i trasy HTTP czytają to samo jezioro danych: to, co czytają trasy, "
        "i to, co dostaje wiersz poleceń, musi być jedną wartością.",
    ),
    "test_api_path_safety.py::a_preflight_from_another_site_is_not_granted": (
        "da39a3ee",
        "Zapytanie wstępne (preflight) z innej strony nie dostaje zgody.",
    ),
    "test_api_path_safety.py::a_request_naming_another_host_is_refused": (
        "f9f99dc1",
        "Żądanie z nagłówkiem Host wskazującym inną maszynę jest odrzucane: atak DNS rebinding "
        "sprawia, że wroga domena wygląda na tę samą stronę, ale zdradza ją nagłówek Host.",
    ),
    "test_api_path_safety.py::delete_refuses_a_run_that_names_another_directory": (
        "da39a3ee",
        "Usuwanie odrzuca przebieg, którego nazwa wskazuje inny katalog.",
    ),
    "test_api_path_safety.py::delete_refuses_an_unknown_model_or_interval": (
        "da39a3ee",
        "Usuwanie odrzuca nieznany model albo interwał.",
    ),
    "test_api_path_safety.py::delete_still_removes_a_real_run": (
        "da39a3ee",
        "Usuwanie nadal usuwa istniejący przebieg.",
    ),
    "test_api_path_safety.py::eval_refuses_a_run_identity_that_is_not_one": (
        "da39a3ee",
        "Ocena odrzuca identyfikator przebiegu, który nim nie jest.",
    ),
    "test_api_path_safety.py::require_inside_refuses_a_path_that_resolves_outside": (
        "da39a3ee",
        "Kontrola ścieżki odrzuca ścieżkę, która po rozwinięciu wychodzi poza jezioro danych.",
    ),
    "test_api_path_safety.py::the_dashboard_host_is_still_served": (
        "da39a3ee",
        "Panel pod własnym adresem jest nadal obsługiwany.",
    ),
    "test_api_path_safety.py::the_storage_layer_refuses_a_run_name_that_is_a_path": (
        "b1eba7a0",
        "Warstwa zapisu odrzuca nazwę przebiegu, która jest ścieżką. To ostatnia linia obrony: "
        "kod, który zapomni o walidacji, i tak nie zbuduje takiej ścieżki.",
    ),
    "test_api_path_safety.py::train_refuses_an_interval_that_is_not_one": (
        "da39a3ee",
        "Trening odrzuca wartość, która nie jest interwałem.",
    ),
    "test_api_path_safety.py::train_renders_numeric_parameters_itself": (
        "da39a3ee",
        "Trening sam formatuje parametry liczbowe polecenia.",
    ),
    "test_api_performance.py::an_untruncated_response_says_so": (
        "da39a3ee",
        "Odpowiedź, której nie obcięto, mówi o tym wprost.",
    ),
    "test_api_performance.py::chart_scan_prunes_partitions_outside_the_range": (
        "da39a3ee",
        "Odczyt danych wykresu pomija partycje spoza żądanego zakresu.",
    ),
    "test_api_performance.py::evaluation_detail_filters_by_flag_server_side": (
        "da39a3ee",
        "Szczegóły oceny filtrowane są według flagi jakości po stronie serwera.",
    ),
    "test_api_performance.py::evaluation_detail_is_bounded_by_limit": (
        "da39a3ee",
        "Szczegóły oceny są ograniczone limitem wierszy.",
    ),
    "test_api_performance.py::evaluation_detail_rejects_an_absurd_limit": (
        "da39a3ee",
        "Szczegóły oceny odrzucają absurdalny limit.",
    ),
    "test_api_performance.py::partition_pruning_keeps_unparseable_names": (
        "14626289",
        "Pomijanie partycji zachowuje te o nieczytelnej nazwie: błędna nazwa partycji nigdy "
        "nie może ukryć danych.",
    ),
    "test_api_performance.py::prediction_responses_are_capped": (
        "5daf3e7b",
        "Odpowiedzi z prognozami mają górny limit, bo cały przebieg nie może trafić do jednej "
        "odpowiedzi: wszystkie świece przebiegu 1d dawały 5 MB, a przebieg 1h z tego samego "
        "okresu jest 24 razy większy. Trasa nie przyjmowała żadnego limitu, więc jedyną "
        "ochroną był opis proszący o podanie roku i miesiąca.",
    ),
    "test_api_performance.py::task_log_buffer_is_bounded_and_cursor_stays_absolute": (
        "da39a3ee",
        "Bufor dziennika zadań ma ograniczony rozmiar, a kursor odczytu pozostaje bezwzględny.",
    ),
    "test_api_performance.py::the_cap_cannot_be_raised_from_the_query_string": (
        "da39a3ee",
        "Limitu nie da się podnieść parametrem zapytania.",
    ),
    "test_api_performance.py::worst_predictions_are_sorted_by_error": (
        "da39a3ee",
        "Najgorsze prognozy są posortowane według błędu.",
    ),
    "test_api_pool_command.py::a_malformed_symbol_never_reaches_argv": (
        "1aa2e58e",
        "Błędny symbol nigdy nie trafia do argumentów polecenia: ten napis staje się "
        "argumentem wiersza poleceń, więc jest walidowany tak samo jak --symbol, a nie "
        "przyjmowany na wiarę.",
    ),
    "test_api_pool_command.py::an_unknown_horizon_name_is_dropped_from_the_override": (
        "da39a3ee",
        "Nieznana nazwa horyzontu jest pomijana w nadpisaniu okien.",
    ),
    "test_api_pool_command.py::no_pool_flag_when_the_caller_named_no_symbols": (
        "da39a3ee",
        "Gdy wywołujący nie wskazał symboli, flaga puli nie jest wysyłana.",
    ),
    "test_api_pool_command.py::the_pool_never_reaches_a_predictor_without_the_flag": (
        "12d78e15",
        "Pula nie trafia do modelu, który nie ma tej flagi - taki przebieg zakończyłby się "
        "błędem, zamiast ją zignorować.",
    ),
    "test_api_pool_command.py::the_pool_reaches_the_tabular_predictors": (
        "da39a3ee",
        "Pula trafia do modeli tabelarycznych.",
    ),
    "test_api_pool_command.py::the_predicted_symbol_is_not_repeated_into_its_own_pool": (
        "da39a3ee",
        "Symbol prognozowany nie jest powtarzany we własnej puli.",
    ),
    "test_api_pool_command.py::the_window_override_is_omitted_unless_chosen": (
        "37b103f7",
        "Nadpisanie szerokości okna jest pomijane, jeśli nie zostało wybrane: szerokość wynika "
        "z horyzontu i historii, a wartość wysłana bez prośby zastąpiłaby decyzję podjętą na "
        "podstawie danych.",
    ),
    "test_api_server.py::build_aggregate_command_requires_an_interval": (
        "da39a3ee",
        "Polecenie agregacji wymaga interwału.",
    ),
    "test_api_server.py::build_pipeline_command_covers_every_selected_interval": (
        "da39a3ee",
        "Polecenie potoku obejmuje każdy wybrany interwał.",
    ),
    "test_api_server.py::build_train_command_passes_interval_and_research_flags": (
        "da39a3ee",
        "Polecenie treningu przekazuje interwał i flagi trybu badawczego.",
    ),
    "test_api_server.py::command_builders_reject_a_bad_symbol": (
        "da39a3ee",
        "Budowanie poleceń odrzuca błędny symbol.",
    ),
    "test_api_server.py::delete_symbol_data_removes_feature_and_label_partitions": (
        "da39a3ee",
        "Usunięcie danych symbolu usuwa także partycje cech i etykiet.",
    ),
    "test_api_server.py::get_kline_chart_data_returns_lightweight_payload": (
        "da39a3ee",
        "Dane świec do wykresu zwracane są w lekkiej postaci.",
    ),
    "test_api_server.py::get_symbol_metadata_returns_base_and_quote_assets": (
        "da39a3ee",
        "Metadane symboli zawierają waluty bazowe i kwotowane.",
    ),
    "test_api_server.py::repair_reaggregates_and_rebuilds_only_the_broken_intervals": (
        "da39a3ee",
        "Naprawa agreguje ponownie i przebudowuje tylko uszkodzone interwały.",
    ),
    "test_api_server.py::repair_redownloads_when_the_1m_base_has_gaps": (
        "da39a3ee",
        "Naprawa pobiera dane ponownie, gdy bazowy interwał 1m ma luki.",
    ),
    "test_api_server.py::repair_rejects_an_unknown_symbol": (
        "da39a3ee",
        "Naprawa odrzuca nieznany symbol.",
    ),
    "test_api_server.py::repair_returns_nothing_for_a_healthy_symbol": (
        "da39a3ee",
        "Dla symbolu bez usterek naprawa nie zwraca żadnych kroków.",
    ),
    "test_backtest_execution.py::a_signal_too_small_to_pay_its_costs_is_not_traded": (
        "5dfae0d3",
        "Sygnał zbyt słaby, by pokryć koszty, nie jest wykonywany. Bramka kosztowa działa w "
        "symulacji, a nie tylko w planerze: bezpośrednie wywołanie map_to_recommendation ze "
        "statycznymi wartościami domyślnymi pominęłoby ją razem z kwantylowymi bramkami "
        "ryzyka, bo nie wyznacza oczekiwanego ruchu, i ta sama świeca mogłaby być tu "
        "Accumulate, a w raporcie rekomendacji Hold. Obie ścieżki przechodzą przez "
        "SequenceGrader. Przy przewadze 0,10 ruch ważony prawdopodobieństwem wynosi 0,31% "
        "wobec progu kosztowego 0,4%, więc nic nie powinno zostać kupione.",
    ),
    "test_backtest_execution.py::buy_and_hold_benchmark_is_reported": (
        "c8800cd5",
        "Raportowany jest punkt odniesienia kupna i trzymania: stopa zwrotu bez niego nie "
        "odpowiada na pytanie, czy wynik był lepszy od trzymania.",
    ),
    "test_backtest_execution.py::costs_are_charged_on_traded_notional": (
        "c67c622e",
        "Koszty naliczane są od wartości transakcji: obowiązują zarówno prowizja, jak i "
        "poślizg, i tylko od tego, co faktycznie zmieniło właściciela.",
    ),
    "test_backtest_execution.py::exposure_scales_with_conviction": (
        "0c1e4feb",
        "Ekspozycja rośnie z siłą sygnału: słaby sygnał nie może angażować całego kapitału.",
    ),
    "test_backtest_execution.py::metrics_cover_risk_not_just_return": (
        "da39a3ee",
        "Miary obejmują ryzyko, a nie tylko stopę zwrotu.",
    ),
    "test_backtest_execution.py::signals_execute_on_the_next_bar_not_their_own": (
        "80ece85e",
        "Sygnał wykonywany jest na następnej świecy, a nie na własnej: sygnał ze świecy t nie "
        "może zostać zrealizowany po jej zamknięciu. Szereg skacze drugiego dnia; realizacja "
        "sygnału z pierwszego dnia po jego zamknięciu przechwyciłaby skok za darmo, a "
        "realizacja po otwarciu drugiego dnia nie może tego zrobić.",
    ),
    "test_backtest_execution.py::the_simulation_and_the_planner_agree_on_every_bar": (
        "bb83e7df",
        "Symulacja i planer zgadzają się co do każdej świecy: jeden oceniacz sprawia, że "
        "raport i symulacja nie mogą opisywać różnych transakcji - to ten defekt doprowadził "
        "do powstania SequenceGrader.",
    ),
    "test_backtest_execution.py::tiny_drift_does_not_trigger_a_rebalance": (
        "2aff74fe",
        "Drobne odchylenie nie wywołuje zmiany pozycji: handel na szumie przepalałby prowizje "
        "bez powodu.",
    ),
    "test_backtest_execution.py::untrained_rows_are_not_traded": (
        "da39a3ee",
        "Wiersze nietrenowane nie są przedmiotem handlu.",
    ),
    "test_backtest_nulls.py::a_constant_exposure_path_reads_as_no_information": (
        "1e0f3688",
        "Stała ścieżka ekspozycji odczytywana jest jako brak informacji. Każde przesunięcie "
        "stałej ścieżki jest tą samą ścieżką; liczenie wyłącznie ścisłych zwycięstw dawało "
        "percentyl 0, czyli najgorsze możliwe wyczucie momentu, choć ścieżka bez zmienności "
        "nie niesie żadnego. Remisy liczą się za pół, więc wynik wynosi 50.",
    ),
    "test_backtest_nulls.py::a_series_too_short_for_the_lookback_yields_nothing": (
        "da39a3ee",
        "Szereg krótszy od okna reguły momentum nie daje wyniku.",
    ),
    "test_backtest_nulls.py::a_series_too_short_to_shift_is_reported_as_such": (
        "da39a3ee",
        "Szereg zbyt krótki do przesunięć jest raportowany jako taki.",
    ),
    "test_backtest_nulls.py::a_wipeout_stops_at_total_loss": (
        "fb8d7d37",
        "Strata zatrzymuje się na utracie całego kapitału: dźwignia nie jest modelowana, więc "
        "kapitał nie może stać się ujemny.",
    ),
    "test_backtest_nulls.py::full_exposure_with_no_costs_reproduces_the_price_return": (
        "da39a3ee",
        "Pełna ekspozycja bez kosztów odtwarza zwrot z ceny.",
    ),
    "test_backtest_nulls.py::inverted_foresight_lands_at_the_bottom": (
        "da39a3ee",
        "Odwrócona znajomość przyszłości wypada na samym dole rozkładu przesunięć.",
    ),
    "test_backtest_nulls.py::perfect_foresight_beats_its_own_mistimed_copies": (
        "0635c924",
        "Doskonała znajomość przyszłości bije własne przesunięte kopie: test musi umieć wykryć "
        "umiejętność, inaczej nie wykrywa niczego.",
    ),
    "test_backtest_nulls.py::the_api_forwards_the_whole_comparison_set": (
        "7acdd90f",
        "Interfejs HTTP przekazuje komplet punktów odniesienia: trasa buduje odpowiedź pole po "
        "polu, więc nowe pole łatwo pominąć, a punktu odniesienia, którego przeglądarka nie "
        "dostaje, nikt nie przeczyta.",
    ),
    "test_backtest_nulls.py::the_backtest_tab_shows_every_comparison": (
        "da39a3ee",
        "Zakładka symulacji pokazuje każdy punkt odniesienia.",
    ),
    "test_backtest_nulls.py::the_momentum_rule_contains_no_fitted_parameter": (
        "12255caf",
        "Reguła momentum nie zawiera dopasowywanych parametrów: jej jedynym ustawieniem jest "
        "stała, a nie widzi ani etykiety, ani prognozy.",
    ),
    "test_backtest_nulls.py::the_momentum_rule_holds_a_rising_market_and_nothing_else": (
        "da39a3ee",
        "Reguła momentum trzyma pozycję wyłącznie na rynku rosnącym.",
    ),
    "test_backtest_nulls.py::the_ui_only_uses_css_classes_that_exist": (
        "a9ff84fe",
        "Panel używa wyłącznie istniejących klas CSS: text-yellow i table-scroll wymyślono w "
        "pierwszej wersji i wyświetliłyby się bez stylu - po cichu, widoczne tylko dla "
        "oglądającego.",
    ),
    "test_backtest_nulls.py::turnover_is_charged_on_every_change": (
        "da39a3ee",
        "Obrót obciążany jest kosztem przy każdej zmianie pozycji.",
    ),
    "test_backtest_nulls.py::zero_exposure_earns_nothing": (
        "da39a3ee",
        "Zerowa ekspozycja nic nie zarabia.",
    ),
    "test_backtest_parameters.py::a_bad_policy_is_refused_rather_than_ignored": (
        "6185a9fa",
        "Błędna polityka ekspozycji jest odrzucana, a nie ignorowana: ciche pominięcie "
        "zostawiłoby wywołującego w przekonaniu, że założenie zastosowano - ten rodzaj błędu "
        "sprawił, że panel uczył 5 epok, a wiersz poleceń 20.",
    ),
    "test_backtest_parameters.py::a_partial_policy_merges_onto_the_default": (
        "da39a3ee",
        "Częściowa polityka ekspozycji uzupełnia domyślną.",
    ),
    "test_backtest_parameters.py::display_names_are_accepted_too": (
        "da39a3ee",
        "Przyjmowane są także wyświetlane nazwy stopni rekomendacji.",
    ),
    "test_backtest_parameters.py::every_assumption_is_a_parameter_of_the_simulation": (
        "da39a3ee",
        "Każde założenie wykonawcze jest parametrem symulacji.",
    ),
    "test_backtest_parameters.py::hold_cannot_be_given_a_target": (
        "4b262626",
        "Stopniowi Hold nie można nadać docelowej ekspozycji: oznacza on pozostawienie pozycji "
        "bez zmian, czyli brak celu, a nie zero. Liczba w tym miejscu zamieniłaby neutralną "
        "prognozę w sygnał.",
    ),
    "test_backtest_parameters.py::no_leverage_and_no_shorts_can_be_expressed": (
        "160e1062",
        "Nie da się wyrazić dźwigni ani krótkiej sprzedaży: na rynku SPOT, przy pozycjach "
        "wyłącznie długich, każdy udział pozostaje ułamkiem kapitału.",
    ),
    "test_backtest_parameters.py::no_policy_means_the_default_is_used_untouched": (
        "da39a3ee",
        "Brak polityki oznacza użycie domyślnej bez zmian.",
    ),
    "test_backtest_parameters.py::the_api_forwards_every_one_of_them": (
        "dd7fc148",
        "Interfejs HTTP przekazuje każde z założeń: trasa buduje odpowiedź i wywołanie "
        "ręcznie, więc pominięty parametr po cichu wróciłby do wartości domyślnej.",
    ),
    "test_backtest_parameters.py::the_assumptions_panel_is_laid_out_and_sized_by_its_content": (
        "7ea136cd",
        "Panel założeń ma układ i wysokość wynikające z zawartości. Jego pola leżą w "
        ".evaluation-controls, które potrzebuje własnej reguły - bez niej każda etykieta "
        "zajmuje pełną szerokość, a lista wychodzi poza kartę. Panel nie może też dziedziczyć "
        "stałej wysokości .ops-card: zamknięty zostawiałby puste pole, a otwarty przycinałby "
        "pola.",
    ),
    "test_backtest_parameters.py::the_browser_only_sends_what_was_changed": (
        "95c15564",
        "Przeglądarka wysyła tylko zmienione pola: wysyłanie wszystkich przy każdym wywołaniu "
        "bez powodu odróżniałoby żądanie od zwykłego wywołania z wiersza poleceń i oddawałoby "
        "przeglądarce wartości domyślne, które do niej nie należą.",
    ),
    "test_backtest_parameters.py::the_cli_offers_the_same_set": (
        "da39a3ee",
        "Wiersz poleceń udostępnia ten sam zestaw założeń.",
    ),
    "test_backtest_parameters.py::the_defaults_endpoint_reads_the_real_constants": (
        "da39a3ee",
        "Trasa wartości domyślnych czyta rzeczywiste stałe.",
    ),
    "test_backtest_parameters.py::the_form_carries_no_hardcoded_assumption": (
        "393b912b",
        "Formularz nie zawiera założeń wpisanych na stałe: liczba w znaczniku jest drugą "
        "definicją, a pole epok kosztowało już ten projekt przebieg uczony 5 epok przy "
        "deklarowanych 20.",
    ),
    "test_backtest_parameters.py::the_reset_control_is_wired": (
        "da39a3ee",
        "Przycisk przywracania wartości domyślnych jest podłączony.",
    ),
    "test_calibration.py::a_coin_flip_earns_no_nll_points": (
        "555d0135",
        "Rzut monetą nie dostaje punktów za stratę logarytmiczną: składnik nagradza pokonanie "
        "niewiedzy, a nie jej osiągnięcie.",
    ),
    "test_calibration.py::a_degenerate_vector_falls_back_to_uniform": (
        "da39a3ee",
        "Zdegenerowany wektor prawdopodobieństw zastępowany jest rozkładem jednostajnym.",
    ),
    "test_calibration.py::a_forecast_without_skill_can_be_flattened_to_a_coin_flip": (
        "a1cb1ba0",
        "Prognozę bez umiejętności da się spłaszczyć do rzutu monetą. Surowa pewność modeli "
        "sięga 0,80-0,96, a przy siatce kończącej się na 10 prognoza 0,99 zostawała na 0,61; "
        "temperatura stawała tam w trzech miesiącach na cztery.",
    ),
    "test_calibration.py::a_legacy_three_class_run_keeps_its_own_reference": (
        "da39a3ee",
        "Starszy przebieg trzyklasowy zachowuje własny poziom odniesienia.",
    ),
    "test_calibration.py::a_missing_model_yields_the_identity": (
        "da39a3ee",
        "Brak modelu daje kalibrację tożsamościową.",
    ),
    "test_calibration.py::a_model_is_calibrated_on_its_own_holdout_predictions": (
        "da39a3ee",
        "Model kalibrowany jest na własnych prognozach dla danych odłożonych.",
    ),
    "test_calibration.py::a_model_that_cannot_score_the_holdout_is_not_fatal": (
        "364b4410",
        "Model, który nie potrafi ocenić danych odłożonych, nie przerywa pracy: kalibracja "
        "jest ulepszeniem, nigdy warunkiem prognozy.",
    ),
    "test_calibration.py::a_region_too_small_to_hold_out_is_left_whole": (
        "528045b0",
        "Obszar zbyt mały, by wydzielić dane odłożone, pozostaje w całości: poniżej progu "
        "liczby próbek kalibrator dopasowywałby szum, więc jest pomijany.",
    ),
    "test_calibration.py::a_single_class_column_is_placed_at_its_own_index": (
        "f6c800ca",
        "Kolumna jedynej klasy trafia na własną pozycję: okno, w którym wystąpiła tylko klasa "
        "long, nadal trafia do właściwej kolumny.",
    ),
    "test_calibration.py::a_single_class_holdout_falls_back_to_the_raw_loss": (
        "b6ef0206",
        "Dane odłożone z jedną klasą wracają do surowej straty: minimalizacja po T na jednej "
        "klasie sprowadza stratę do zera za darmo.",
    ),
    "test_calibration.py::a_single_class_holdout_never_sharpens": (
        "6d6eab05",
        "Dane odłożone z jedną klasą nigdy nie wyostrzają prognozy. Minimalizacja straty "
        "logarytmicznej na jednej klasie sprowadza temperaturę do dolnej granicy: model ma "
        "rację na każdym odłożonym wierszu, więc strata maleje wraz z temperaturą, a "
        "dopasowanie zamienia prognozę zbyt pewną w pewną. Na BTCUSDT horyzont 365-dniowy "
        "trafia na to często - w długich okresach każdy zwrot w przód ma ten sam znak - i "
        "zostawiał długi horyzont z |przewagą| = 1,000 na każdej świecy.",
    ),
    "test_calibration.py::a_zeroed_class_does_not_produce_a_nan": (
        "da39a3ee",
        "Wyzerowana klasa nie daje wartości NaN.",
    ),
    "test_calibration.py::an_empty_holdout_is_infinitely_bad_not_zero": (
        "65a229ed",
        "Puste dane odłożone mają stratę nieskończoną, a nie zerową: zero sprawiłoby, że epoka "
        "niemożliwa do oceny wyglądałaby na najlepszą.",
    ),
    "test_calibration.py::apply_to_mapping_keeps_the_class_names": (
        "da39a3ee",
        "Kalibracja słownika prawdopodobieństw zachowuje nazwy klas.",
    ),
    "test_calibration.py::better_separation_lowers_the_calibrated_loss": (
        "a7a752af",
        "Lepsze rozdzielenie klas obniża stratę po kalibracji: miara nadal coś mierzy - "
        "pozostaje w niej zdolność rozróżniania.",
    ),
    "test_calibration.py::calibration_never_reaches_past_the_training_bound": (
        "b11f1c0a",
        "Kalibracja nigdy nie sięga za granicę treningu: wycinek odłożony musi być historią "
        "już zrealizowaną. training_bounds cofa koniec okna o forward_steps od prognozowanej "
        "świecy, więc obie części podziału leżą ściśle przed nią.",
    ),
    "test_calibration.py::calibration_preserves_the_ranking_for_every_temperature": (
        "da39a3ee",
        "Kalibracja zachowuje kolejność klas przy każdej temperaturze.",
    ),
    "test_calibration.py::fitting_leaves_an_already_calibrated_forecast_alone": (
        "da39a3ee",
        "Dopasowanie nie zmienia prognozy już skalibrowanej.",
    ),
    "test_calibration.py::fitting_recovers_a_softening_temperature_from_overconfidence": (
        "3e3af320",
        "Dopasowanie odzyskuje temperaturę zmiękczającą prognozę zbyt pewną: prognozy 0,95, "
        "trafne mniej więcej w połowie przypadków, wymagają T > 1.",
    ),
    "test_calibration.py::max_entropy_follows_the_class_count": (
        "e7a557a6",
        "Maksymalna entropia zależy od liczby klas. Normalizowanie straty logarytmicznej przez "
        "ln(3) po przejściu na dwie klasy dawało rzutowi monetą (strata ln 2 = 0,693, "
        "podzielona przez ln 3 = 1,099) wynik 0,37 umiejętności, której nie powinien dostać "
        "wcale, więc każdy przebieg w tabeli dostawał punkty za zgadywanie.",
    ),
    "test_calibration.py::sharpening_raises_the_raw_loss_but_not_the_calibrated_one": (
        "14e544e5",
        "Wyostrzenie podnosi surową stratę, ale nie stratę po kalibracji. To dlatego uczenie "
        "sieci kończyło się po jednej epoce: surowa entropia krzyżowa karze za pewność, więc "
        "sieć zachowująca ten sam porządek, lecz wyrażająca go mocniej, wypada gorzej, choć "
        "dopasowanie temperatury po treningu cofa dokładnie to. Na BTCUSDT surowa strata "
        "wynosiła kolejno 0,7244, 0,9109, 0,9282 i 1,0216 - rosła od pierwszej epoki - podczas "
        "gdy trafność wciąż rosła.",
    ),
    "test_calibration.py::temperature_above_one_softens_an_overconfident_forecast": (
        "da39a3ee",
        "Temperatura powyżej 1 zmiękcza prognozę zbyt pewną.",
    ),
    "test_calibration.py::temperature_below_one_sharpens": (
        "da39a3ee",
        "Temperatura poniżej 1 wyostrza prognozę.",
    ),
    "test_calibration.py::the_calibrated_loss_never_exceeds_the_raw_one": (
        "902f8732",
        "Strata po kalibracji nigdy nie przekracza surowej: T = 1 należy do siatki, więc "
        "minimum może być tylko niższe albo równe.",
    ),
    "test_calibration.py::the_holdout_is_carved_out_of_the_training_region": (
        "da39a3ee",
        "Dane odłożone wydzielane są z obszaru treningowego.",
    ),
    "test_calibration.py::the_identity_temperature_changes_nothing": (
        "da39a3ee",
        "Temperatura tożsamościowa niczego nie zmienia.",
    ),
    "test_calibration.py::too_few_samples_yields_the_identity": (
        "da39a3ee",
        "Zbyt mała liczba próbek daje kalibrację tożsamościową.",
    ),
    "test_calibration_record.py::gradient_boosting_records_what_calibration_did": (
        "da39a3ee",
        "Model scikit-learn zapisuje przy prognozie temperaturę, jej źródło i "
        "prawdopodobieństwa sprzed kalibracji.",
    ),
    "test_calibration_record.py::gru_records_what_calibration_did": (
        "da39a3ee",
        "GRU zapisuje przy prognozie temperaturę, jej źródło i prawdopodobieństwa sprzed "
        "kalibracji.",
    ),
    "test_calibration_record.py::tft_records_what_calibration_did": (
        "da39a3ee",
        "TFT zapisuje przy prognozie temperaturę, jej źródło i prawdopodobieństwa sprzed "
        "kalibracji.",
    ),
    "test_calibration_record.py::xgboost_records_what_calibration_did": (
        "da39a3ee",
        "XGBoost zapisuje przy prognozie temperaturę, jej źródło i prawdopodobieństwa sprzed "
        "kalibracji.",
    ),
    "test_cli_api_contract.py::a_card_with_an_open_section_grows_with_it": (
        "b64982e2",
        "Karta z rozwiniętą sekcją rośnie razem z nią: .ops-card ma stałą wysokość, aby moduły "
        "w rzędzie się wyrównywały, a rozwinięte sekcje formularza treningu przy stałej "
        "wysokości wychodziłyby poza kartę, na to, co leży niżej.",
    ),
    "test_cli_api_contract.py::a_horizon_subset_reaches_the_command": (
        "da39a3ee",
        "Wybrany podzbiór horyzontów trafia do polecenia.",
    ),
    "test_cli_api_contract.py::an_explicit_epoch_count_is_passed_through": (
        "da39a3ee",
        "Jawnie podana liczba epok jest przekazywana dalej.",
    ),
    "test_cli_api_contract.py::cli_epoch_defaults_come_from_the_shared_module": (
        "da39a3ee",
        "Domyślne liczby epok w wierszu poleceń pochodzą ze wspólnego modułu.",
    ),
    "test_cli_api_contract.py::every_predictor_accepts_a_horizon_selection": (
        "da39a3ee",
        "Każdy model przyjmuje wybór horyzontów.",
    ),
    "test_cli_api_contract.py::every_predictor_accepts_what_the_api_sends": (
        "da39a3ee",
        "Każdy model przyjmuje to, co wysyła interfejs HTTP.",
    ),
    "test_cli_api_contract.py::every_predictor_takes_an_interval": (
        "67c817e0",
        "Każdy model przyjmuje interwał: horyzonty są rozpiętościami kalendarzowymi, więc "
        "interwał nigdy nie jest domyślny.",
    ),
    "test_cli_api_contract.py::opening_a_runs_details_selects_that_run": (
        "140a8708",
        "Otwarcie szczegółów przebiegu wybiera ten przebieg: karty pokazują przebieg otwarty z "
        "rankingu, a „Evaluate Run” i wykres stabilności działają na wyborze obok, więc "
        "kliknięcie „Details” musi przestawić wybór na przebieg pokazany w kartach.",
    ),
    "test_cli_api_contract.py::research_flags_reach_every_predictor": (
        "6cae38fb",
        "Flagi trybu badawczego trafiają do każdego modelu: --deterministic i --seed są "
        "dostępne we wszystkich.",
    ),
    "test_cli_api_contract.py::the_api_does_not_impose_an_epoch_count_the_caller_never_chose": (
        "7965bb81",
        "Interfejs HTTP nie narzuca liczby epok, której wywołujący nie wybrał. Inaczej panel "
        "uczy 5 epok tam, gdzie wiersz poleceń uczy 20: wartość domyślna Typer obowiązuje "
        "tylko przy braku flagi, więc interfejs dopisujący --epochs przy każdym wywołaniu "
        "zawsze narzucałby własną wartość - dwie definicje jednej wartości, z których jedna po "
        "cichu wygrywa.",
    ),
    "test_cli_api_contract.py::the_api_never_invents_a_model_type": (
        "d04e5377",
        "Interfejs HTTP nigdy nie wymyśla rodzaju modelu: o tym, które modele istnieją, "
        "decyduje wiersz poleceń.",
    ),
    "test_cli_api_contract.py::the_calendar_title_is_in_the_interface_language": (
        "26d044ce",
        "Tytuł kalendarza jest w języku interfejsu: interfejs jest angielski, a nazwa miesiąca "
        "w języku przeglądarki umieszczałaby „czerwiec 2026” nad angielskimi nazwami dni "
        "tygodnia.",
    ),
    "test_cli_api_contract.py::the_dead_walkforward_knobs_are_gone": (
        "52262727",
        "Martwe ustawienia pętli kroczącej zostały usunięte: train_months był przyjmowany i "
        "ignorowany, bo nic tu w ten sposób nie douczało modelu.",
    ),
    "test_cli_api_contract.py::the_form_exposes_a_horizon_control": (
        "da39a3ee",
        "Formularz udostępnia wybór horyzontów.",
    ),
    "test_cli_api_contract.py::the_full_horizon_set_is_not_spelled_out": (
        "99cd679d",
        "Pełny zbiór horyzontów nie jest wypisywany: wysyłanie wszystkich przy każdym "
        "wywołaniu bez powodu odróżniałoby argumenty od zwykłego wywołania z wiersza poleceń.",
    ),
    "test_cli_api_contract.py::the_prediction_calendar_names_the_horizon_it_shows": (
        "00a9620e",
        "Kalendarz prognoz nazywa pokazywany horyzont: komórka pokazuje najkrótszy horyzont, "
        "więc nosi jego nazwę. Rozpiętość wpisana w szablon („4w”) zostawała po zmianie zbioru "
        "horyzontów i opisywała tydzień jako cztery tygodnie.",
    ),
    "test_cli_api_contract.py::the_sharpness_bar_does_not_hardcode_three_classes": (
        "2583a8c1",
        "Pasek ostrości prognozy nie zakłada trzech klas: kierunek ma dwie klasy, a ln(3) "
        "dawało rzutowi monetą 0,37 ostrości, której nie powinien mieć wcale - ten sam defekt "
        "co wcześniej po stronie Pythona.",
    ),
    "test_cli_api_contract.py::the_stability_action_is_wired_to_a_handler": (
        "360f3dc5",
        "Akcja stabilności jest podłączona do obsługi: atrybuty data-action obsługuje jeden "
        "delegowany nasłuch, a atrybut bez wpisu w rejestrze jest martwym przyciskiem.",
    ),
    "test_cli_api_contract.py::the_stability_command_takes_full_iso_dates": (
        "73549d13",
        "Polecenie stabilności przyjmuje pełne daty ISO: jego poprzednik przyjmował tu "
        "RRRR-MM, a bliźniacza symulacja RRRR-MM-DD. Dwa sąsiednie polecenia nad tym samym "
        "przebiegiem, które nie zgadzają się co do formatu daty, podważają zaufanie do wiersza "
        "poleceń opisanego w załączniku pracy.",
    ),
    "test_cli_api_contract.py::the_training_form_has_no_hardcoded_epoch_value": (
        "b12b8805",
        "Formularz treningu nie ma wpisanej na stałe liczby epok: miał value=\"5\", podczas "
        "gdy wiersz poleceń domyślnie uczył 20. Liczba musi pochodzić z "
        "/api/meta/training-defaults, aby formularz i wiersz poleceń nie rozjechały się "
        "ponownie.",
    ),
    "test_cli_api_contract.py::the_ui_exposes_the_stability_panel": (
        "a88f8abe",
        "Panel udostępnia raport stabilności: raportu dostępnego tylko z wiersza poleceń nikt "
        "nie czyta.",
    ),
    "test_cli_api_contract.py::the_ui_offers_every_model_the_api_accepts": (
        "4ca29684",
        "Panel oferuje każdy model przyjmowany przez interfejs HTTP: modelu odniesienia "
        "brakowało na liście, więc nie dało się go wytrenować z panelu, choć przyjmował go "
        "interfejs, implementował wiersz poleceń, a każde porównanie od niego zależy.",
    ),
    "test_cli_exit_codes.py::a_deliberate_exit_is_not_reported_twice": (
        "70ef9dd9",
        "Celowe zakończenie nie jest zgłaszane dwukrotnie: ogólna obsługa wyjątków łapała "
        "typer.Exit(1) zgłoszony w bloku try i po właściwym komunikacie wypisywała drugi "
        "wiersz „Error: 1”.",
    ),
    "test_cli_exit_codes.py::a_failure_keeps_its_traceback": (
        "f4c70a0c",
        "Błąd zachowuje ślad stosu: polecenia wypisywały tylko komunikat wyjątku, więc "
        "nieoczekiwany błąd był jednym wierszem bez miejsca wystąpienia - i tylko tyle "
        "pokazywał dziennik zadań w panelu.",
    ),
    "test_cli_exit_codes.py::aggregate_all_failed_exits_one": (
        "da39a3ee",
        "Agregacja, w której nic się nie powiodło, kończy się kodem 1.",
    ),
    "test_cli_exit_codes.py::aggregate_no_work_no_errors_exits_zero": (
        "da39a3ee",
        "Agregacja bez pracy do wykonania i bez błędów kończy się kodem 0.",
    ),
    "test_cli_exit_codes.py::aggregate_partial_errors_exits_zero": (
        "da39a3ee",
        "Agregacja z częściowymi błędami kończy się kodem 0.",
    ),
    "test_cli_exit_codes.py::backfill_partial_success_exits_zero": (
        "da39a3ee",
        "Pobieranie z częściowym sukcesem kończy się kodem 0.",
    ),
    "test_cli_exit_codes.py::backfill_zero_processed_exits_one": (
        "da39a3ee",
        "Pobieranie, które nie przetworzyło niczego, kończy się kodem 1.",
    ),
    "test_cost_threshold.py::a_confident_but_tiny_move_does_not_clear_costs": (
        "da39a3ee",
        "Pewny, ale drobny ruch nie pokrywa kosztów.",
    ),
    "test_cost_threshold.py::a_large_expected_move_clears_costs": (
        "da39a3ee",
        "Duży oczekiwany ruch pokrywa koszty.",
    ),
    "test_cost_threshold.py::a_legacy_run_is_measured_on_its_own_bins": (
        "cae7bfc0",
        "Starszy przebieg mierzony jest na własnych kubełkach: kubełka 34+ nie ma w obecnym "
        "schemacie, ale nadal musi mieć wagę.",
    ),
    "test_cost_threshold.py::an_empty_distribution_yields_no_move": (
        "da39a3ee",
        "Pusty rozkład nie daje ruchu.",
    ),
    "test_cost_threshold.py::an_uncleared_signal_becomes_hold_however_strong": (
        "da39a3ee",
        "Sygnał, który nie pokrywa kosztów, staje się Hold niezależnie od siły.",
    ),
    "test_cost_threshold.py::depth_decides_where_edge_cannot": (
        "4a702812",
        "Głębokość rozstrzyga tam, gdzie przewaga nie może: te same prawdopodobieństwa "
        "kierunku, przeciwne werdykty - na tym polega bramka.",
    ),
    "test_cost_threshold.py::the_cost_is_configurable_per_market": (
        "da39a3ee",
        "Koszt można ustawić osobno dla każdego rynku.",
    ),
    "test_cost_threshold.py::the_gate_defaults_open": (
        "b6d9a54f",
        "Bramka jest domyślnie otwarta: wywołania bez rozkładu głębokości oceniane są "
        "wyłącznie na podstawie przewagi i ryzyka.",
    ),
    "test_cost_threshold.py::the_gate_is_symmetric": (
        "d55b18ce",
        "Bramka jest symetryczna: wyjście kosztuje tyle co wejście, a niezmiennik 8 musi "
        "przetrwać bramkę.",
    ),
    "test_cost_threshold.py::the_open_top_bin_still_contributes": (
        "9002be48",
        "Otwarty górny kubełek nadal się liczy: 144+ nie ma środka przedziału i nie może po "
        "cichu liczyć się jako zero.",
    ),
    "test_data_manager.py::chart_data_joins_features_labels_and_detects_gaps": (
        "da39a3ee",
        "Dane wykresu łączą cechy z etykietami i wykrywają luki.",
    ),
    "test_data_manager.py::inventory_reports_intervals_quality_and_cache_refresh": (
        "da39a3ee",
        "Inwentarz raportuje interwały, jakość danych i odświeżenie pamięci podręcznej.",
    ),
    "test_data_manager.py::missing_labels_are_reported_for_every_interval": (
        "039fb5c3",
        "Brak etykiet raportowany jest dla każdego interwału: sprawdzenie opóźnienia dla "
        "interwałów innych niż 1m nie może kończyć łańcucha stanów, inaczej interwał 1d bez "
        "żadnych etykiet zostaje zgłoszony jako kompletny.",
    ),
    "test_depth_scheme.py::a_run_scheme_is_read_from_its_own_payload": (
        "da39a3ee",
        "Schemat przebiegu odczytywany jest z jego własnych danych.",
    ),
    "test_depth_scheme.py::large_move_bins_are_recognised_in_both_schemes": (
        "da39a3ee",
        "Kubełki dużych ruchów rozpoznawane są w obu schematach.",
    ),
    "test_depth_scheme.py::lower_edge_is_read_from_the_label": (
        "da39a3ee",
        "Dolna granica kubełka odczytywana jest z jego etykiety.",
    ),
    "test_depth_scheme.py::only_domain_defines_the_bin_scheme": (
        "da39a3ee",
        "Schemat kubełków definiuje wyłącznie moduł domenowy.",
    ),
    "test_depth_scheme.py::risk_is_unchanged_by_splitting_the_tail": (
        "fc910ea9",
        "Podział ogona nie zmienia ryzyka: rozbicie 34+ na cztery kubełki dzieli masę "
        "prawdopodobieństwa, ale nie może jej przesunąć.",
    ),
    "test_depth_scheme.py::settings_take_their_bins_from_the_domain": (
        "da39a3ee",
        "Konfiguracja bierze kubełki z modułu domenowego.",
    ),
    "test_depth_scheme.py::small_bins_are_not_large": (
        "da39a3ee",
        "Małe kubełki nie są uznawane za duże.",
    ),
    "test_depth_scheme.py::the_label_list_matches_the_scheme_string": (
        "da39a3ee",
        "Lista etykiet zgadza się z zapisem schematu.",
    ),
    "test_depth_scheme.py::the_open_bin_gets_a_representative_magnitude": (
        "da39a3ee",
        "Otwarty kubełek dostaje reprezentatywną wielkość ruchu.",
    ),
    "test_depth_scheme.py::the_scheme_is_ordered_by_magnitude_not_alphabetically": (
        "558811fc",
        "Schemat uporządkowany jest według wielkości, a nie alfabetycznie: '8-13' jako napis "
        "wypada po '144+', a kolejność musi być liczbowa.",
    ),
    "test_depth_scheme.py::the_tail_reaches_144": (
        "d9670114",
        "Ogon sięga 144%: roczny ruch BTC ma medianę wielkości około 59%. Przy 34+ jako górnym "
        "kubełku trafiałoby do niego 70% etykiet długich horyzontów, a prognoza głębokości "
        "zdegenerowałaby się do stałej.",
    ),
    "test_download_verification.py::a_corrupted_archive_on_disk_is_fetched_again": (
        "da39a3ee",
        "Uszkodzone archiwum na dysku jest pobierane ponownie.",
    ),
    "test_download_verification.py::a_dry_run_fetches_nothing": (
        "da39a3ee",
        "Tryb próbny niczego nie pobiera.",
    ),
    "test_download_verification.py::a_matching_archive_is_kept": (
        "da39a3ee",
        "Archiwum o zgodnej sumie kontrolnej jest zachowywane.",
    ),
    "test_download_verification.py::a_mismatched_archive_is_removed_and_refused": (
        "da39a3ee",
        "Archiwum o niezgodnej sumie kontrolnej jest usuwane i odrzucane.",
    ),
    "test_download_verification.py::a_verified_archive_on_disk_is_not_fetched_again": (
        "da39a3ee",
        "Sprawdzone archiwum na dysku nie jest pobierane ponownie.",
    ),
    "test_download_verification.py::an_unreadable_checksum_is_a_failed_check_not_a_skipped_one": (
        "da39a3ee",
        "Nieczytelna suma kontrolna oznacza nieudaną kontrolę, a nie pominiętą.",
    ),
    "test_early_stopping.py::a_holdout_too_small_to_read_spends_the_budget": (
        "264b0bb3",
        "Przy zbyt małym zbiorze odłożonym uczenie wykorzystuje cały limit epok: poniżej progu "
        "sygnał zatrzymania jest szumem, więc nie jest używany.",
    ),
    "test_early_stopping.py::a_nearby_split_rescues_a_single_class_fit_half": (
        "0bb984da",
        "Pobliski punkt podziału ratuje część uczącą z jedną klasą. Starsza część z jedną "
        "klasą nie może kosztować okna danych odłożonych: calibration_split dzieli według "
        "pozycji i nie widzi etykiet, więc na długim horyzoncie - gdzie rok dat startowych "
        "może mieć jeden znak - starsza część często zawiera jedną klasę. Modelu nie da się na "
        "niej dopasować, a rozwiązanie zastępcze (uczenie na wszystkim, bez danych odłożonych) "
        "po cichu wyłącza w tym oknie zarówno wczesne zatrzymanie, jak i kalibrację.",
    ),
    "test_early_stopping.py::a_run_records_its_own_epoch_usage": (
        "497103ee",
        "Przebieg zapisuje liczbę wykorzystanych epok: to wynik przebiegu, więc trafia obok "
        "prognoz. Pozostawiona w słowniku wyników zginęłaby razem z procesem - wiersz poleceń "
        "wypisuje dwa inne pola - a pytanie, czy wczesne zatrzymanie zadziałało, wymagałoby za "
        "każdym razem osobnego sprawdzenia.",
    ),
    "test_early_stopping.py::a_single_class_window_cannot_be_rescued": (
        "da39a3ee",
        "Okna z jedną klasą nie da się uratować.",
    ),
    "test_early_stopping.py::a_single_transition_cannot_be_rescued": (
        "1d4f1132",
        "Pojedynczej zmiany klasy nie da się uratować: przy jednej zmianie żaden podział nie "
        "zostawia obu części mieszanych - to arytmetyka.",
    ),
    "test_early_stopping.py::a_zero_ceiling_cannot_be_hit": (
        "8545c5c5",
        "Zerowego limitu nie da się osiągnąć: chroni to przed liczeniem każdego okna jako "
        "ograniczonego limitem, gdy limitu brak, co wyglądałoby tak, jakby wczesne zatrzymanie "
        "nigdy nie działało.",
    ),
    "test_early_stopping.py::an_already_valid_split_is_left_alone": (
        "da39a3ee",
        "Poprawny już podział pozostaje bez zmian.",
    ),
    "test_early_stopping.py::both_deep_predictors_persist_their_summary": (
        "79e3c8c0",
        "Oba modele głębokie zapisują swoje podsumowanie: pola widocznego tylko w konsoli nikt "
        "nie czyta.",
    ),
    "test_early_stopping.py::epoch_usage_is_reported_against_its_ceiling": (
        "dc16765a",
        "Wykorzystanie epok raportowane jest wobec limitu: --epochs jest górną granicą, więc "
        "wykorzystana liczba jest wynikiem przebiegu i pokazuje, czy wczesne zatrzymanie w "
        "ogóle zadziałało.",
    ),
    "test_early_stopping.py::per_horizon_budgets_override_the_global_one": (
        "da39a3ee",
        "Limity epok dla horyzontów zastępują limit ogólny.",
    ),
    "test_early_stopping.py::the_best_weights_are_restored_not_the_last": (
        "f13ed53c",
        "Przywracane są najlepsze wagi, a nie ostatnie: zatrzymanie bez przywrócenia oddaje "
        "najgorszy model okna.",
    ),
    "test_early_stopping.py::the_budget_is_still_a_hard_ceiling": (
        "da39a3ee",
        "Limit epok pozostaje twardą granicą.",
    ),
    "test_early_stopping.py::the_rescue_never_returns_a_single_class_holdout": (
        "aa787550",
        "Ratowanie podziału nigdy nie zwraca danych odłożonych z jedną klasą: entropię "
        "krzyżową na jednej klasie minimalizuje prognoza tej klasy z pewnością, więc "
        "zatrzymanie na niej nagradzałoby nadmierną pewność, którą kalibrator ma usuwać.",
    ),
    "test_early_stopping.py::the_rescue_stays_close_to_the_requested_fraction": (
        "da39a3ee",
        "Ratowanie podziału trzyma się blisko żądanego udziału danych odłożonych.",
    ),
    "test_early_stopping.py::training_stops_before_the_budget_when_the_holdout_stalls": (
        "da39a3ee",
        "Uczenie zatrzymuje się przed limitem, gdy strata na danych odłożonych przestaje "
        "maleć.",
    ),
    "test_eval_stability.py::a_class_the_run_never_emitted_scores_as_a_miss": (
        "da39a3ee",
        "Klasa, której przebieg nigdy nie wskazał, liczy się jako pomyłka.",
    ),
    "test_eval_stability.py::a_regime_change_is_visible_month_by_month": (
        "da39a3ee",
        "Zmiana reżimu rynku jest widoczna miesiąc po miesiącu.",
    ),
    "test_eval_stability.py::a_thin_month_is_flagged_rather_than_dropped": (
        "58e0ed97",
        "Miesiąc z małą liczbą świec jest oznaczany, a nie pomijany: średnia z trzech świec "
        "zmienia się z powodów niezwiązanych z modelem, ale ukrycie miesiąca sprawiłoby, że "
        "szereg wyglądałby na gęstszy, niż jest.",
    ),
    "test_eval_stability.py::an_exactly_zero_return_has_no_side": (
        "fb204e2a",
        "Zwrot dokładnie zerowy nie ma strony: nie może zostać wciśnięty w klasę, więc wiersz "
        "jest wykluczany.",
    ),
    "test_eval_stability.py::an_hourly_run_is_scored_bar_by_bar_not_date_by_date": (
        "1db20e54",
        "Przebieg godzinowy oceniany jest świeca po świecy, a nie data po dacie (niezmiennik "
        "4). Poprzednik indeksował zrealizowane zwroty datą kalendarzową i czytał je z szeregu "
        "1d, więc wszystkie 24 świece godzinowe dnia miały jeden wynik, a horyzont liczono od "
        "dziennego zamknięcia zamiast od świecy prognozy.",
    ),
    "test_eval_stability.py::an_untrained_horizon_contributes_nothing": (
        "ac6754ae",
        "Horyzont nietrenowany nic nie wnosi (niezmiennik 3): rozkład jednostajny nie jest "
        "prognozą.",
    ),
    "test_eval_stability.py::dates_are_full_iso_like_every_sibling_command": (
        "a0fb3d54",
        "Daty mają pełny format ISO (RRRR-MM-DD), jak w każdym sąsiednim poleceniu.",
    ),
    "test_eval_stability.py::direction_is_the_sign_of_the_return": (
        "da39a3ee",
        "Kierunek jest znakiem zwrotu.",
    ),
    "test_eval_stability.py::it_does_not_need_the_planner_to_have_run": (
        "ede0a227",
        "Ocena stabilności nie wymaga wcześniejszego uruchomienia planera: usunięcie wyniku "
        "finansowego usunęło razem z nim zależność od rekomendacji.",
    ),
    "test_eval_stability.py::multiclass_logloss_brier_are_finite_positive": (
        "da39a3ee",
        "Strata logarytmiczna i wynik Briera dla wielu klas są skończone i dodatnie.",
    ),
    "test_eval_stability.py::scoring_follows_the_run_s_own_class_set": (
        "9303da8c",
        "Punktacja korzysta z własnego zbioru klas przebiegu: starszy przebieg trzyklasowy "
        "pozostaje oceniany tą samą funkcją.",
    ),
    "test_eval_stability.py::the_artifact_is_written_under_the_source_run": (
        "da39a3ee",
        "Artefakt zapisywany jest pod przebiegiem źródłowym.",
    ),
    "test_eval_stability.py::the_bar_key_normalises_every_open_time_shape": (
        "da39a3ee",
        "Klucz świecy normalizuje każdą postać czasu otwarcia.",
    ),
    "test_eval_stability.py::the_horizon_is_converted_to_the_run_s_own_interval": (
        "f85346ca",
        "Horyzont przeliczany jest na interwał przebiegu (niezmiennik 2): long musi oznaczać "
        "rok także przy 1h, a nie 365 godzin. Przy 90 dniach historii żadna świeca nie ma "
        "zrealizowanego zwrotu rocznego, więc długi horyzont musi pozostać nieoceniony; "
        "potraktowanie forward_days jako liczby świec szukałoby wyniku 365 godzin naprzód i "
        "oceniało prawie każdą świecę.",
    ),
    "test_eval_stability.py::the_report_carries_no_returns": (
        "23447d85",
        "Raport nie zawiera stóp zwrotu: wynik finansowy należy do symulacji, która liczy go "
        "poprawnie, a ten raport go powielał, z inną polityką ekspozycji.",
    ),
    "test_features_build.py::build_features_exports_professional_analysis_indicators": (
        "da39a3ee",
        "Wyznaczanie cech eksportuje wskaźniki analizy technicznej.",
    ),
    "test_features_build.py::build_features_uses_daily_warmup_before_requested_start": (
        "da39a3ee",
        "Wyznaczanie cech korzysta z dni rozruchowych sprzed żądanego początku.",
    ),
    "test_features_build.py::partitions_with_and_without_order_flow_build_together": (
        "e88d14ed",
        "Partycje z kolumnami przepływu zleceń i bez nich budują się razem: miesiące zapisane "
        "przed dodaniem tych kolumn zawierają tylko OHLCV, nowsze - pięć kolumn więcej. Zwykłe "
        "łączenie odrzucało taką mieszankę błędem ShapeError, co zatrzymało przebudowę cech "
        "ETHUSDT dla interwału 12h.",
    ),
    "test_forward_labels.py::a_horizon_is_the_same_span_at_every_interval": (
        "96e9439a",
        "Horyzont oznacza tę samą rozpiętość przy każdym interwale: przy 1h etykieta "
        "tygodniowa patrzy 168 świec naprzód, a nie 7.",
    ),
    "test_forward_labels.py::a_one_bar_label_file_counts_as_outdated": (
        "3f546169",
        "Plik etykiet z jedną świecą uznawany jest za nieaktualny: to prawdziwy plik parquet o "
        "złej zawartości.",
    ),
    "test_forward_labels.py::matches_the_loop_it_replaced": (
        "da39a3ee",
        "Wynik zgadza się z pętlą, którą zastąpiła wspólna definicja.",
    ),
    "test_forward_labels.py::nan_has_no_side": (
        "982413c1",
        "Wartość NaN nie ma strony: nie spełnia ani == 0, ani > 0, więc bez sprawdzenia "
        "trafiałaby do klasy short.",
    ),
    "test_forward_labels.py::no_predictor_carries_its_own_copy": (
        "da39a3ee",
        "Żaden model nie ma własnej kopii definicji etykiety.",
    ),
    "test_forward_labels.py::the_chart_shows_the_traded_horizon": (
        "da39a3ee",
        "Wykres pokazuje horyzont, na którym prowadzony jest handel.",
    ),
    "test_forward_labels.py::the_file_holds_every_horizon_the_models_train": (
        "da39a3ee",
        "Plik etykiet zawiera każdy horyzont, na którym uczą się modele.",
    ),
    "test_hash_utils.py::read_checksum_file": (
        "1a4eb478",
        "Odczyt sumy kontrolnej z pliku CHECKSUM.",
    ),
    "test_hash_utils.py::read_checksum_file_not_found": (
        "1d2b8b12",
        "Odczyt nieistniejącego pliku sumy kontrolnej.",
    ),
    "test_hash_utils.py::sha256_file": ("d8371d2f", "Wyznaczanie skrótu SHA256 pliku."),
    "test_hash_utils.py::sha256_file_not_found": (
        "c02059b2",
        "Wyznaczanie skrótu SHA256 nieistniejącego pliku.",
    ),
    "test_hash_utils.py::verify_checksum": ("92ba37ab", "Weryfikacja sumy kontrolnej."),
    "test_history_shorter_than_range.py::a_window_of_one_bar_still_normalises": (
        "bb7fe867",
        "Okno z jedną świecą nadal daje się znormalizować: zabezpieczenie ma działać tylko "
        "przy naprawdę pustym wejściu; jeden wiersz jest przypadkiem zdegenerowanym, ale "
        "dobrze określonym, a dolna granica rozstępu międzykwartylowego obsługuje zerowy "
        "rozrzut.",
    ),
    "test_history_shorter_than_range.py::an_empty_fitting_window_is_refused_not_crashed": (
        "88b6e2cf",
        "Puste okno dopasowania jest odrzucane, a nie przerywa pracy: None oznacza „nie da się "
        "znormalizować”, co wywołujący zamienia na świecę nietrenowaną. Zwrócenie zer dawałoby "
        "modelowi po cichu zmyśloną skalę.",
    ),
    "test_history_shorter_than_range.py::both_sequence_predictors_refuse_the_empty_window": (
        "6a0f40f5",
        "Obie sieci normalizują tą samą funkcją, więc obie miały ten sam błąd; puste okno daje "
        "teraz świecę nietrenowaną.",
    ),
    "test_history_shorter_than_range.py::normalisation_reads_only_the_window_it_is_given": (
        "c0e822c6",
        "Normalizacja czyta wyłącznie przekazane jej okno: dopasowuje się do świec przeszłych "
        "i stosuje do wszystkich, a gdyby statystyki pochodziły z całej tablicy, prognoza "
        "widziałaby własną przyszłość.",
    ),
    "test_holdout_purge.py::a_gap_that_leaves_nothing_to_fit_holds_nothing_out": (
        "a0f40c9c",
        "Horyzont roczny w oknie kilkumiesięcznym: uczenie na całym oknie bez kalibratora jest "
        "lepsze niż kalibrator bez modelu.",
    ),
    "test_holdout_purge.py::a_moved_pooled_split_still_starts_with_a_whole_instant": (
        "da39a3ee",
        "Przesunięty podział puli nadal zaczyna wycinek od pełnej chwili.",
    ),
    "test_holdout_purge.py::a_pooled_instant_is_never_on_both_sides": (
        "5288410d",
        "Trzy instrumenty na dzień: wiersze jednego dnia to ten sam rynek, więc dzień nie leży "
        "po obu stronach podziału, a horyzont jest odcinkiem czasu, nie liczbą wierszy.",
    ),
    "test_holdout_purge.py::a_rescued_split_is_judged_on_the_rows_actually_fitted": (
        "2934412c",
        "Przesunięty podział oceniany jest na wierszach rzeczywiście uczących: bez przerwy "
        "najstarsze 100 wierszy wygląda na zdatne, z przerwą uczy się tylko 70, a one mają "
        "jedną klasę.",
    ),
    "test_holdout_purge.py::a_run_never_leaves_a_trained_window_uncalibrated": (
        "da39a3ee",
        "Przebieg nie zostawia żadnego wytrenowanego okna bez kalibracji.",
    ),
    "test_holdout_purge.py::a_window_without_a_slice_takes_the_horizons_latest_temperature": (
        "85d91e2b",
        "Okno bez wycinka przejmuje ostatnią temperaturę horyzontu, wyznaczoną na wycinku "
        "zrealizowanym wcześniej, a przed pierwszą nie dostaje pewności ponad losową. Okno bez "
        "kalibracji było najpewniejsze w przebiegu tylko dlatego, że brakowało danych.",
    ),
    "test_holdout_purge.py::gradient_boosting_calibrates_on_labels_it_was_not_fitted_on": (
        "da39a3ee",
        "Model scikit-learn kalibruje się na etykietach, na których się nie uczył.",
    ),
    "test_holdout_purge.py::gru_stops_and_calibrates_on_labels_it_was_not_fitted_on": (
        "da39a3ee",
        "GRU zatrzymuje uczenie i kalibruje się na etykietach, na których się nie uczył.",
    ),
    "test_holdout_purge.py::tft_stops_and_calibrates_on_labels_it_was_not_fitted_on": (
        "da39a3ee",
        "TFT zatrzymuje uczenie i kalibruje się na etykietach, na których się nie uczył.",
    ),
    "test_holdout_purge.py::tft_stops_on_the_calibration_slice_alone": (
        "3d706ebb",
        "TFT decyduje o zatrzymaniu wyłącznie na wycinku kalibracyjnym; świece sprzed niego są "
        "tylko historią kodera, nigdy celem, bo model był na nich uczony albo leżą w przerwie.",
    ),
    "test_holdout_purge.py::the_fitted_rows_end_a_horizon_before_the_held_out_slice": (
        "da39a3ee",
        "Wiersze uczące kończą się jeden horyzont przed wycinkiem kalibracyjnym.",
    ),
    "test_holdout_purge.py::the_policy_holds_nothing_out_when_no_split_works": (
        "da39a3ee",
        "Gdy żaden podział nie spełnia warunków, okno nie wydziela wycinka.",
    ),
    "test_holdout_purge.py::the_policy_keeps_the_default_split_when_it_is_usable": (
        "da39a3ee",
        "Zasada podziału zostawia podział domyślny, gdy jest dopuszczalny.",
    ),
    "test_holdout_purge.py::the_policy_moves_a_split_whose_slice_holds_one_class": (
        "ff64cd18",
        "Zasada podziału przesuwa podział, którego wycinek ma jedną klasę: zatrzymanie i "
        "kalibracja na jednej klasie nagradzają pewność.",
    ),
    "test_holdout_purge.py::tree_models_move_a_split_that_leaves_one_class_to_fit": (
        "3b816984",
        "Modele drzewiaste przesuwają podział zostawiający do uczenia jedną klasę, tak jak "
        "sieci; wcześniej na tym samym oknie jedna architektura była skalibrowana, a druga nie.",
    ),
    "test_holdout_purge.py::xgboost_calibrates_on_labels_it_was_not_fitted_on": (
        "da39a3ee",
        "XGBoost kalibruje się na etykietach, na których się nie uczył.",
    ),
    "test_horizon_set.py::a_quarter_carries_more_than_twice_a_half_year": (
        "a8be9581",
        "Kwartał niesie ponad dwa razy więcej obserwacji niezależnych niż półrocze - to "
        "zmierzony powód wyboru kwartału, a nie preferencja.",
    ),
    "test_horizon_set.py::a_run_with_unknown_horizons_fails_loudly": (
        "8d339941",
        "Przebieg z nieznanymi horyzontami kończy się jawnym błędem: starszy przebieg nie może "
        "zostać oceniony na zerowej liczbie wierszy i wyglądać jak model, który nic nie "
        "prognozował. Brakujący horyzont trafiał do normalize_probabilities({}), które zwraca "
        "rozkład jednostajny, więc oceniano rozkład bazowy jak prognozę - czemu zapobiega "
        "niezmiennik 3.",
    ),
    "test_horizon_set.py::every_horizon_is_a_whole_number_of_weeks": (
        "a598800e",
        "Każdy horyzont jest pełną liczbą tygodni: days_to_steps zaokrągla, więc inna wartość "
        "oznaczałaby, że interwał 1w zadaje inne pytanie niż 1d, czego zabrania niezmiennik 2.",
    ),
    "test_horizon_set.py::the_conversion_is_exact_on_every_commensurate_interval": (
        "da39a3ee",
        "Przeliczenie jest dokładne na każdym współmiernym interwale.",
    ),
    "test_horizon_set.py::the_names_do_not_collide_with_the_direction_classes": (
        "87278239",
        "Nazwy horyzontów nie kolidują z klasami kierunku: short i long były wcześniej zarówno "
        "horyzontami, jak i kierunkami.",
    ),
    "test_horizon_set.py::the_set_is_ordered_shortest_first": (
        "b2312877",
        "Zbiór uporządkowany jest od najkrótszego horyzontu: kolejność ma znaczenie, bo "
        "komórka kalendarza, domyślny horyzont handlu i paleta wykresów biorą pierwszy element "
        "jako najkrótszy.",
    ),
    "test_horizon_set.py::the_shorter_horizons_are_the_well_powered_ones": (
        "b7407baa",
        "Krótsze horyzonty mają wystarczającą moc statystyczną - dlatego dodano tydzień, a rok "
        "raportowany jest jako niemierzalny, zamiast z dokładnością do trzech miejsc po "
        "przecinku.",
    ),
    "test_horizon_set.py::the_spans_are_commensurate": (
        "da39a3ee",
        "Rozpiętości horyzontów są współmierne.",
    ),
    "test_integration.py::generate_month_range": ("b730767f", "Generowanie zakresu miesięcy."),
    "test_integration.py::process_month_dry_run": (
        "4de7cb3c",
        "Przetworzenie miesiąca w trybie próbnym niczego nie pobiera.",
    ),
    "test_integration.py::storage_layout_paths": (
        "f1bd40f8",
        "Ścieżki układu katalogów są spójne.",
    ),
    "test_inventory_cache.py::a_stale_cache_is_not_served_after_data_is_lost": (
        "1cf554f7",
        "Po utracie danych nieaktualna pamięć podręczna nie jest zwracana: zwykłe wywołanie "
        "inwentarza, bez wymuszania odświeżenia, musi zobaczyć stratę.",
    ),
    "test_inventory_cache.py::an_untouched_lake_still_hits_the_cache": (
        "72e6c653",
        "Niezmienione jezioro danych nadal korzysta z pamięci podręcznej: odcisk musi być "
        "stabilny, inaczej pamięć podręczna nie ma sensu.",
    ),
    "test_inventory_cache.py::deleting_a_symbol_removes_every_interval": (
        "f8457145",
        "Usunięcie symbolu usuwa każdy interwał: _symbol_paths obejmowało tylko cechy i "
        "etykiety 1d, więc pozostałe interwały przetrwały usunięcie symbolu, a ponieważ "
        "inwentarz opiera się na świecach, resztki były potem niewidoczne. Ta sama lista "
        "mierzy symbol, więc zaniżała też raportowany rozmiar.",
    ),
    "test_inventory_cache.py::fingerprint_changes_when_a_month_is_added": (
        "da39a3ee",
        "Odcisk zmienia się po dodaniu miesiąca.",
    ),
    "test_inventory_cache.py::fingerprint_changes_when_a_month_is_deleted": (
        "da39a3ee",
        "Odcisk zmienia się po usunięciu miesiąca.",
    ),
    "test_inventory_cache.py::fingerprint_changes_when_a_whole_year_is_deleted": (
        "da39a3ee",
        "Odcisk zmienia się po usunięciu całego roku.",
    ),
    "test_inventory_cache.py::symbol_size_counts_every_interval": (
        "da39a3ee",
        "Rozmiar symbolu obejmuje każdy interwał.",
    ),
    "test_labels.py::assign_depth_bin": ("49a82b0c", "Przypisanie wartości do kubełka głębokości."),
    "test_labels.py::parse_depth_bins": ("b2104c40", "Odczyt zapisu kubełków głębokości."),
    "test_labels.py::parse_depth_bins_invalid": (
        "c3a2b107",
        "Odczyt błędnego zapisu kubełków głębokości.",
    ),
    "test_missing_features.py::a_missing_feature_lands_at_the_window_median": (
        "da39a3ee",
        "Brakująca cecha trafia na medianę okna.",
    ),
    "test_missing_features.py::both_networks_scale_the_same_window_identically": (
        "da39a3ee",
        "Obie sieci skalują to samo okno identycznie.",
    ),
    "test_missing_features.py::the_scale_is_taken_from_the_values_present": (
        "46c24ab1",
        "Skala liczona jest z wartości obecnych: mediana 0 i rozstęp międzykwartylowy 1 na "
        "świecach, które mają cechę.",
    ),
    "test_one_grader.py::a_bar_is_graded_before_its_risk_joins_the_history": (
        "da39a3ee",
        "Świeca oceniana jest, zanim jej ryzyko dołączy do historii.",
    ),
    "test_one_grader.py::grading_never_returns_something_outside_the_scale": (
        "da39a3ee",
        "Ocena nigdy nie zwraca wartości spoza skali rekomendacji.",
    ),
    "test_one_grader.py::no_expected_move_leaves_the_cost_gate_open": (
        "776f842f",
        "Brak oczekiwanego ruchu zostawia bramkę kosztową otwartą: wywołanie bez rozkładu "
        "głębokości nie może sprowadzić wszystkiego do Hold.",
    ),
    "test_one_grader.py::nothing_new_grades_a_bar_on_its_own": (
        "da39a3ee",
        "Żaden nowy kod nie ocenia świecy samodzielnie.",
    ),
    "test_one_grader.py::the_cost_gate_holds_a_move_too_small_to_pay_for_itself": (
        "da39a3ee",
        "Bramka kosztowa zatrzymuje ruch zbyt mały, by się opłacić.",
    ),
    "test_one_grader.py::the_grader_is_symmetric": (
        "3630ce58",
        "Oceniacz jest symetryczny: niezmiennik 13 musi przetrwać także opakowanie.",
    ),
    "test_one_grader.py::the_grader_starts_on_the_historical_constants": (
        "a7aecb75",
        "Oceniacz zaczyna od stałych historycznych: rozruch musi przebiegać tak, jak opisano, "
        "inaczej krótki przebieg po cichu oceniałby świece względem kwantyli z niemal pustej "
        "historii.",
    ),
    "test_order_flow_features.py::aggregation_sums_the_order_flow_columns": (
        "35be9f7b",
        "Agregacja sumuje kolumny przepływu zleceń: kod sumujący istniał, ale nigdy nie mógł "
        "zadziałać, bo kolumny do niego nie docierały.",
    ),
    "test_order_flow_features.py::empty_input_keeps_the_minimal_schema": (
        "da39a3ee",
        "Puste wejście zachowuje minimalny schemat.",
    ),
    "test_order_flow_features.py::features_still_build_without_the_order_flow_columns": (
        "00aea4cd",
        "Cechy budują się także bez kolumn przepływu zleceń: świece zapisane przed "
        "rozszerzeniem schematu muszą nadal działać.",
    ),
    "test_order_flow_features.py::full_kline_columns_survive_conversion": (
        "da39a3ee",
        "Wszystkie kolumny świecy przetrwają konwersję.",
    ),
    "test_order_flow_features.py::ohlcv_only_sources_still_work": (
        "da39a3ee",
        "Źródła zawierające tylko OHLCV nadal działają.",
    ),
    "test_order_flow_features.py::order_flow_features_appear_when_the_columns_exist": (
        "da39a3ee",
        "Cechy przepływu zleceń pojawiają się, gdy istnieją odpowiednie kolumny.",
    ),
    "test_planner_windows.py::calculate_edge": ("da39a3ee", "Wyznaczanie przewagi prognozy."),
    "test_planner_windows.py::calculate_risk_metric_counts_only_the_adverse_side": (
        "68577283",
        "Miara ryzyka liczy wyłącznie stronę niekorzystną: prognozie wzrostu zagraża spadek, a "
        "nie jej własny wzrost.",
    ),
    "test_planner_windows.py::conviction_the_risk_does_not_support_falls_back_to_hold": (
        "da39a3ee",
        "Przekonanie, którego ryzyko nie uzasadnia, sprowadzane jest do Hold.",
    ),
    "test_planner_windows.py::every_step_of_the_scale_is_reachable": (
        "5d03e81d",
        "Każdy stopień skali jest osiągalny: każdą ocenę, łącznie ze Strong Sell, daje jakaś "
        "para (przewaga, ryzyko). Przy kolejności od najsłabszego Reduce przesłaniałoby Sell i "
        "Strong Sell, bo każde ryzyko dość niskie dla silnego sygnału spełnia też słabszą "
        "gałąź, sprawdzaną wcześniej.",
    ),
    "test_planner_windows.py::map_to_recommendation_valid_domain": (
        "da39a3ee",
        "Odwzorowanie na rekomendację działa w całej dziedzinie.",
    ),
    "test_planner_windows.py::the_scale_is_symmetric_around_zero_edge": (
        "da39a3ee",
        "Skala jest symetryczna wokół zerowej przewagi.",
    ),
    "test_pooled_training.py::a_summary_reports_effective_symbols_beside_the_row_count": (
        "c0d90df8",
        "Podsumowanie podaje liczbę symboli efektywnych obok liczby wierszy: wiersze zawyżają "
        "korzyść z łączenia trzykrotnie, a podsumowanie z samymi wierszami wspierałoby "
        "twierdzenie, którego dane nie potwierdzają.",
    ),
    "test_pooled_training.py::a_symbol_that_had_not_launched_contributes_nothing": (
        "eb8f300a",
        "Symbol, którego jeszcze nie notowano, nic nie wnosi. To pułapka okna opartego na "
        "indeksie: wiersz 400 SOLUSDT przypada na rok 2021, więc okno indeksowe wyśrodkowane "
        "na roku 2018 wciągnęłoby przyszłe świece w przeszłość.",
    ),
    "test_pooled_training.py::an_empty_pool_is_refused": ("da39a3ee", "Pusta pula jest odrzucana."),
    "test_pooled_training.py::labels_stay_inside_their_own_symbol": (
        "a7c7ca8c",
        "Etykiety pozostają w obrębie własnego symbolu: etykieta BTCUSDT to znak zwrotu "
        "BTCUSDT. Łączenie dzieli cechy i wyniki, nigdy ścieżki cen - zamknięcie jednego "
        "symbolu decydujące o etykiecie innego byłoby przeciekiem niewidocznym w żadnej "
        "mierze.",
    ),
    "test_pooled_training.py::no_pooled_row_reaches_the_cutoff": (
        "05d07a4b",
        "Żaden wiersz z puli nie sięga granicy treningu: granica jest chwilą w czasie, więc "
        "obowiązuje wszystkie symbole naraz.",
    ),
    "test_pooled_training.py::one_symbol_pooled_matches_the_index_arithmetic_it_replaced": (
        "0b62abcd",
        "Pula z jednym symbolem daje to samo co zastąpiona arytmetyka indeksów - dlatego "
        "przypadek jednego symbolu też przechodzi przez pulę. Na ciągłym szeregu obie metody "
        "muszą wybrać dokładnie te same wiersze, a tam, gdzie mogą się różnić - przy luce w "
        "danych - poprawna jest wersja oparta na znacznikach czasu, bo pyta, czy wynik już "
        "nastąpił, a nie, ile wierszy wstecz leży.",
    ),
    "test_pooled_training.py::pooling_more_symbols_cannot_beat_one_over_the_correlation": (
        "4c4ec9a0",
        "Łączenie większej liczby symboli nie przekroczy 1/ρ - dlatego horyzont roczny nie "
        "jest problemem łączenia. Wyrażenie n / (1 + (n − 1)ρ) dąży do 1/ρ, więc przy "
        "zmierzonym 0,76 nawet sto kryptowalut jest wartych około 1,3 symbolu: dzielą jeden "
        "dominujący czynnik rynkowy, a łączenie uśrednia to, co się różni, a nie to, co "
        "dominuje. Usuwa go dopiero prognoza ruchów względnych.",
    ),
    "test_pooled_training.py::the_correlation_is_measured_from_the_pool_itself": (
        "0ce0b498",
        "Korelacja mierzona jest na samej puli: dwa symbole napędzane tym samym szeregiem "
        "muszą wyjść skorelowane, a dwa niezależne - nie.",
    ),
    "test_pooled_training.py::the_frame_keeps_the_zone_the_parquet_had": (
        "e2cebe8b",
        "Ramka zachowuje strefę czasową z pliku parquet: przejście znaczników czasu przez "
        "datetime64 usuwa +00:00 z każdego open_time, a prognozy są kluczowane przez open_time "
        "(niezmiennik 4). Taki przebieg wygląda na zupełnie zdrowy - po prostu nie łączy się z "
        "niczym zapisanym ze strefą.",
    ),
    "test_pooled_training.py::the_pooled_slice_is_chronological_not_stacked_by_symbol": (
        "89a043f8",
        "Wycinek z puli jest chronologiczny, a nie ułożony symbolami jeden po drugim: "
        "calibration_split bierze najnowsze wiersze obszaru treningowego, a sklejenie symboli "
        "kolejno oddałoby mu najnowszą historię jednego symbolu jako wycinek odłożony.",
    ),
    "test_pooled_training.py::the_predicted_symbol_is_always_in_the_pool_and_first": (
        "8f2a5f13",
        "Symbol prognozowany zawsze jest w puli i to na pierwszym miejscu: przebieg uczony na "
        "wszystkim poza prognozowanym symbolem to inny eksperyment, o który nikt nie prosi, "
        "pomijając flagę.",
    ),
    "test_pooled_training.py::the_slice_is_reproducible_when_symbols_share_a_timestamp": (
        "7d17cf85",
        "Wycinek jest odtwarzalny, gdy symbole mają ten sam znacznik czasu: remisy "
        "rozstrzygane są stabilnie, inaczej dwa przebiegi tej samej konfiguracji mogłyby "
        "podzielić zbiór kalibracyjny inaczej i zgadzać się tylko przypadkiem.",
    ),
    "test_pooled_training.py::three_correlated_symbols_are_not_three_observations": (
        "2c5e2473",
        "Trzy skorelowane symbole to nie trzy obserwacje - ten pomiar rozstrzyga, czy łączenie "
        "w ogóle pomaga. Średnia korelacja par zwrotów w przód na danych projektu wynosi 0,60 "
        "na horyzoncie tygodniowym i 0,76 na rocznym.",
    ),
    "test_predict_interval_integration.py::every_horizon_is_actually_trained": (
        "76055834",
        "Każdy horyzont rzeczywiście się uczy: żaden nie może przez cały przebieg wracać do "
        "nietrenowanego wypełnienia. Chroni to przed zapaścią okna treningowego - przy oknie "
        "węższym od horyzontu dłuższe horyzonty byłyby w całości wypełnieniem, a przebieg "
        "nadal wyglądałby na udany.",
    ),
    "test_predict_interval_integration.py::predict_ml_rejects_window_narrower_than_horizon": (
        "4fdcf5a3",
        "Prognoza odrzuca okno węższe od horyzontu: okno, które nie może wytrenować każdego "
        "horyzontu, jest twardym błędem, a nie cichym zastępstwem.",
    ),
    "test_predict_interval_integration.py::predict_ml_writes_interval_versioned_folder": (
        "da39a3ee",
        "Prognoza zapisuje katalog wersjonowany z interwałem w nazwie.",
    ),
    "test_predict_interval_integration.py::predictions_carry_unique_open_time_per_row": (
        "ef37ff43",
        "Każdy wiersz prognozy ma unikalny czas otwarcia: data powtarza się przy interwałach "
        "krótszych niż dzień, więc prawdziwym kluczem jest open_time.",
    ),
    "test_predict_leakage.py::a_quarter_has_more_than_twice_the_power_of_a_half_year": (
        "77ef005a",
        "Kwartał ma ponad dwa razy większą moc statystyczną niż półrocze - to powód przejścia "
        "na 91 dni, a nie względy estetyczne.",
    ),
    "test_predict_leakage.py::a_sufficient_window_is_accepted": (
        "da39a3ee",
        "Wystarczające okno jest przyjmowane.",
    ),
    "test_predict_leakage.py::a_well_powered_configuration_warns_about_nothing": (
        "da39a3ee",
        "Konfiguracja o wystarczającej mocy nie wywołuje ostrzeżeń.",
    ),
    "test_predict_leakage.py::a_window_narrower_than_the_horizon_is_rejected": (
        "da39a3ee",
        "Okno węższe od horyzontu jest odrzucane.",
    ),
    "test_predict_leakage.py::a_window_needs_headroom_beyond_the_horizon": (
        "da39a3ee",
        "Okno potrzebuje zapasu ponad horyzont.",
    ),
    "test_predict_leakage.py::an_underpowered_configuration_warns_without_refusing": (
        "b11320e2",
        "Konfiguracja o zbyt małej mocy wywołuje ostrzeżenie, ale nie jest odrzucana: horyzont "
        "roczny musi nadal działać - to, że nie da się go zweryfikować, jest wynikiem wartym "
        "raportowania, a nie powodem do ukrycia.",
    ),
    "test_predict_leakage.py::an_underpowered_horizon_is_warned_about": (
        "da39a3ee",
        "Horyzont o zbyt małej mocy wywołuje ostrzeżenie.",
    ),
    "test_predict_leakage.py::horizons_span_the_same_calendar_time_across_intervals": (
        "da39a3ee",
        "Horyzonty obejmują ten sam czas kalendarzowy przy różnych interwałach.",
    ),
    "test_predict_leakage.py::the_longest_horizon_is_a_whole_number_of_the_others": (
        "03b92a4d",
        "Najdłuższy horyzont jest całkowitą wielokrotnością pozostałych: rozpiętości są "
        "współmierne, więc horyzont jest zawsze czystą wielokrotnością krótszego. 365 dni na "
        "to nie pozwala: 365/2 to 182,5, a 365/7 nie jest liczbą całkowitą.",
    ),
    "test_predict_leakage.py::the_requirement_scales_by_interval": (
        "da39a3ee",
        "Wymaganie skaluje się z interwałem.",
    ),
    "test_predict_leakage.py::the_scored_side_is_warned_about_too": (
        "f390e6cb",
        "Ostrzeżenie dotyczy także strony ocenianej: poszerzenie okna, by naprawić stronę "
        "treningową, zabiera świece stronie ocenianej, jedna za jedną. Ostrzeżenie tylko o "
        "pierwszej zachęcałoby do dalszego poszerzania - i tak dobrze wyuczony horyzont staje "
        "się niemierzalny.",
    ),
    "test_predict_leakage.py::the_shipped_horizons_convert_exactly_to_whole_weeks": (
        "78058086",
        "Dostarczone horyzonty przeliczają się dokładnie na pełne tygodnie - dlatego long ma "
        "364 dni, a nie 365. days_to_steps zaokrągla, więc horyzont 365-dniowy stawał się 52 "
        "świecami przy interwale 1w (364 dni), a przy 1d pozostawał 365. Niezmiennik 2 wymaga, "
        "by oba zadawały to samo pytanie, a nie zadawały: long przebiegu 1w był o dzień "
        "krótszy niż przebiegu 1d. Pełne tygodnie całkowicie usuwają zaokrąglenie.",
    ),
    "test_predict_leakage.py::the_two_floors_are_deliberately_different": (
        "e7cf9fdb",
        "Dwa progi są różne: były jedną stałą, wspólną dla porządku, i rozdzieliły je dane. "
        "Próg treningowy pyta, czy model da się tu w ogóle dopasować, co wymaga niewiele. Próg "
        "bootstrapu pyta, czy przedział wokół wyniku jest stabilny, co wymaga o rząd więcej: "
        "przy 3,1 bloku dwa z pięćdziesięciu pięciu przedziałów nie obejmowały własnej oceny "
        "punktowej.",
    ),
    "test_predict_leakage.py::the_window_reports_how_many_observations_each_horizon_has": (
        "dd7a00fd",
        "Okno raportuje, ile obserwacji ma każdy horyzont. resolve_train_windows odrzuca okno, "
        "które nie może trenować; to jego odpowiednik dla okna, które może trenować, ale nie "
        "może mierzyć. Okno 730-dniowe zawiera na horyzoncie rocznym 730 − 364 = 366 świec z "
        "etykietą, które nakładają się w 363 z 364 dni - to około jednej obserwacji "
        "niezależnej. Nie zmieni tego żaden model, optymalizator ani limit epok; to arytmetyka "
        "nienakładającej się historii w dwóch latach i powód, dla którego jedno wspólne okno "
        "musiało zniknąć.",
    ),
    "test_predict_leakage.py::train_window_scales_with_interval": (
        "da39a3ee",
        "Okno treningowe skaluje się z interwałem.",
    ),
    "test_predict_leakage.py::training_bounds_exclude_labels_that_need_future_closes": (
        "da39a3ee",
        "Granice treningu wykluczają etykiety wymagające przyszłych zamknięć.",
    ),
    "test_predict_leakage.py::training_bounds_respects_window_floor": (
        "da39a3ee",
        "Granice treningu respektują minimalną szerokość okna.",
    ),
    "test_predict_leakage.py::untrained_payload_is_flagged_and_normalised": (
        "da39a3ee",
        "Wynik nietrenowany jest oznaczany i normalizowany.",
    ),
    "test_predict_leakage.py::usable_sample_count_is_zero_when_window_matches_horizon": (
        "da39a3ee",
        "Liczba użytecznych próbek wynosi zero, gdy okno równa się horyzontowi.",
    ),
    "test_prediction_evaluation.py::divergences_do_not_depend_on_the_process_hash_seed": (
        "470b5275",
        "Dywergencje nie zależą od ziarna haszowania procesu: dywergencje KL i JS sumowane są "
        "w ustalonej kolejności etykiet. Przy sumie po set(p) | set(q) kolejność zależałaby od "
        "haszowania napisów w danym procesie, a depth_js zmieniałoby się na ostatnim bicie "
        "między procesami.",
    ),
    "test_prediction_evaluation.py::evaluate_predictions_writes_quality_artifacts": (
        "da39a3ee",
        "Ocena prognoz zapisuje artefakty jakości.",
    ),
    "test_prediction_evaluation.py::prediction_metric_functions_are_stable": (
        "da39a3ee",
        "Funkcje miar jakości prognozy są stabilne.",
    ),
    "test_prediction_evaluation.py::sub_daily_predictions_score_against_their_own_bar": (
        "b9c69af7",
        "Prognozy dla interwałów krótszych niż dzień oceniane są na własnej świecy: wiersze "
        "godzinowe dzielą datę kalendarzową, więc rozróżnia je open_time.",
    ),
    "test_prediction_evaluation.py::untrained_horizons_are_excluded_from_metrics": (
        "cdf0e021",
        "Horyzonty nietrenowane są wykluczane z miar: wypełnienie nie może być oceniane jak "
        "prognoza.",
    ),
    "test_predictor_training_edges.py::a_clean_run_reports_no_failures": (
        "da39a3ee",
        "Przebieg bez błędów nie raportuje niepowodzeń.",
    ),
    "test_predictor_training_edges.py::a_gru_window_that_raises_is_counted_and_the_run_goes_on": (
        "efaee659",
        "Okno GRU, które zgłosiło wyjątek, jest liczone, a przebieg trwa dalej: GRU pozwalało "
        "wyjątkowi zakończyć cały przebieg, podczas gdy TFT i XGBoost zapisują takie okno jako "
        "nietrenowane i je liczą. Wiersze traci tylko okno z wyjątkiem - ten sam horyzont uczy "
        "się ponownie w następnym miesiącu.",
    ),
    "test_predictor_training_edges.py::a_window_that_raises_is_counted_not_only_printed": (
        "0e0fedfe",
        "Okno, które zgłosiło wyjątek, jest liczone, a nie tylko wypisywane: nieudane "
        "dopasowanie daje wiersze nietrenowane, co jest poprawne, i jest liczone w "
        "run_summary.json. Gdyby świadczył o nim tylko wiersz w konsoli, przebieg z nieudanymi "
        "miesiącami wyglądałby jak przebieg z krótką historią.",
    ),
    "test_predictor_training_edges.py::baseline_history_does_not_depend_on_the_requested_range": (
        "762b5283",
        "Historia modelu odniesienia nie zależy od żądanego zakresu: okno kroczące patrzy "
        "wstecz, więc nie może być przycinane do [start, koniec]. Wczytanie tylko żądanych "
        "miesięcy zostawiało pierwsze prognozy niemal bez historii, przez co punkt odniesienia "
        "wypadał tym gorzej, im węższy był zakres.",
    ),
    "test_predictor_training_edges.py::xgboost_trains_when_a_window_lacks_one_direction": (
        "8b36084f",
        "XGBoost uczy się także w oknie bez jednego z kierunków: XGBClassifier odrzuca zbiory "
        "etykiet pomijające klasę. Błąd był łapany i zamieniany na horyzont nietrenowany, więc "
        "okno z trendem po cichu wyłączało model, zamiast dopasować go do dostępnych klas.",
    ),
    "test_qa_checks.py::check_duplicates": ("344f3c2e", "Wykrywanie duplikatów."),
    "test_qa_checks.py::check_gaps": ("cd5329fa", "Wykrywanie luk."),
    "test_qa_checks.py::check_ohlc_invariants": ("23c0ade2", "Sprawdzanie niezmienników OHLC."),
    "test_qa_checks.py::check_sorted": ("9d253f03", "Sprawdzanie posortowania."),
    "test_risk_gates.py::a_calm_market_still_gets_a_binding_gate": (
        "82efd3e9",
        "Spokojny rynek nadal dostaje wiążącą bramkę - na tym polegają bramki kwantylowe: "
        "najwyższa ocena pozostaje osiągalna. Każda z tych świec przeszłaby przez stały próg "
        "0,2, więc taka bramka nie wyrażałaby między nimi żadnej preferencji, a wobec własnego "
        "rozkładu spokojniejsza połowa nadal odróżnia się od bardziej rozedrganej.",
    ),
    "test_risk_gates.py::a_short_history_keeps_the_historical_constants": (
        "da39a3ee",
        "Przy krótkiej historii obowiązują stałe historyczne.",
    ),
    "test_risk_gates.py::a_window_does_not_influence_its_own_gate": (
        "64fa79a2",
        "Okno nie wpływa na własną bramkę: bramki są częścią decyzji, więc mogą widzieć "
        "wyłącznie przeszłość. Sprawdzane przez plan_windows, bo to kolejność przenosi tę "
        "własność; test samego risk_gates mógłby tylko powtórzyć, że czysta funkcja ignoruje "
        "argumenty, których nie dostała.",
    ),
    "test_risk_gates.py::custom_gates_keep_the_scale_symmetric": (
        "2677f2d3",
        "Własne bramki zachowują symetrię skali: niezmiennik 13 musi przetrwać nowy parametr.",
    ),
    "test_risk_gates.py::the_default_gates_are_the_historical_constants": (
        "72952cf3",
        "Domyślne bramki to stałe historyczne: wywołanie bez argumentów musi działać dokładnie "
        "jak wcześniej.",
    ),
    "test_risk_gates.py::the_gates_are_recorded_in_the_report": (
        "dd77d3c9",
        "Bramki zapisywane są w raporcie: raport musi być czytelny bez odtwarzania przejścia, "
        "które go oceniło.",
    ),
    "test_risk_gates.py::the_gates_become_quantiles_once_there_is_history": (
        "da39a3ee",
        "Gdy jest historia, bramki stają się kwantylami.",
    ),
    "test_risk_gates.py::the_gates_stay_ordered_for_any_history": (
        "da39a3ee",
        "Bramki pozostają uporządkowane przy dowolnej historii.",
    ),
    "test_risk_gates.py::the_metric_cannot_reach_the_old_outer_gate": (
        "e6a36c10",
        "Miara nie może osiągnąć dawnej zewnętrznej bramki - dlatego stałe musiały zniknąć: "
        "0,4 jest z konstrukcji nieosiągalne. Ryzyko to min(p_long, p_short) · P(duży ruch | "
        "ta strona), a mniejsze z dwóch prawdopodobieństw sumujących się do 1 wynosi co "
        "najwyżej 0,5. Nawet pewny duży ruch przeciwny przy idealnym rzucie monetą daje 0,5, a "
        "każda rzeczywista świeca leży poniżej.",
    ),
    "test_risk_gates.py::tighter_gates_can_only_downgrade": (
        "e286e27b",
        "Ostrzejsze bramki mogą tylko obniżyć ocenę: surowsza bramka nigdy nie podnosi oceny "
        "świecy - obniża ją albo zostawia bez zmian.",
    ),
    "test_risk_is_directional.py::a_bearish_signal_is_not_blocked_by_its_own_downside": (
        "596c6f1e",
        "Sygnał spadkowy nie jest blokowany przez własny spadek: naiwna poprawka "
        "uwzględniająca tylko spadki uczyniłaby Strong Sell nieosiągalnym.",
    ),
    "test_risk_is_directional.py::a_fat_adverse_tail_still_blocks_the_signal": (
        "f9743dcd",
        "Gruby ogon przeciwny nadal blokuje sygnał: bramka musi działać dalej - to przypadek, "
        "dla którego istnieje.",
    ),
    "test_risk_is_directional.py::an_expected_rally_is_not_a_reason_to_avoid_buying": (
        "da39a3ee",
        "Oczekiwany wzrost nie jest powodem, by unikać kupna.",
    ),
    "test_risk_is_directional.py::every_verdict_on_the_scale_is_still_reachable": (
        "da39a3ee",
        "Każdy werdykt na skali pozostaje osiągalny.",
    ),
    "test_risk_is_directional.py::risk_mirrors_when_the_bar_does": (
        "f4589a2b",
        "Ryzyko odbija się lustrzanie razem ze świecą (niezmiennik 8): lustrzane odbicie "
        "świecy musi dać lustrzaną rekomendację.",
    ),
    "test_risk_is_directional.py::the_metric_is_symmetric_across_the_whole_range": (
        "da39a3ee",
        "Miara jest symetryczna w całym zakresie.",
    ),
    "test_row_order.py::every_deduplication_keeps_order_or_sorts_again": (
        "da39a3ee",
        "Każde usuwanie duplikatów zachowuje kolejność albo ponownie sortuje wiersze.",
    ),
    "test_run_attribution.py::backtest_refuses_a_run_with_no_trained_rows": (
        "da39a3ee",
        "Symulacja odrzuca przebieg bez wierszy trenowanych.",
    ),
    "test_run_attribution.py::backtest_reports_the_run_it_simulated": (
        "da39a3ee",
        "Symulacja raportuje przebieg, który symulowała.",
    ),
    "test_run_attribution.py::backtest_without_any_run_explains_itself": (
        "da39a3ee",
        "Symulacja bez żadnego przebiegu wyjaśnia, czego brakuje.",
    ),
    "test_run_attribution.py::list_runs_filters_by_model_and_interval": (
        "da39a3ee",
        "Lista przebiegów filtruje według modelu i interwału.",
    ),
    "test_run_attribution.py::list_runs_returns_newest_first": (
        "da39a3ee",
        "Lista przebiegów zaczyna się od najnowszego.",
    ),
    "test_run_attribution.py::planner_skips_untrained_horizons": (
        "da39a3ee",
        "Planer pomija horyzonty nietrenowane.",
    ),
    "test_run_attribution.py::planner_windows_are_written_under_the_source_run": (
        "da39a3ee",
        "Okna planera zapisywane są pod przebiegiem źródłowym.",
    ),
    "test_run_attribution.py::resolve_run_defaults_to_latest_but_honours_an_explicit_pick": (
        "da39a3ee",
        "Wybór przebiegu domyślnie wskazuje najnowszy, ale respektuje jawny wybór.",
    ),
    "test_run_attribution.py::resolve_run_lists_candidates_when_nothing_matches": (
        "da39a3ee",
        "Gdy nic nie pasuje, wybór przebiegu wypisuje kandydatów.",
    ),
    "test_run_provenance.py::an_unseeded_run_says_so_rather_than_inventing_a_number": (
        "fb70992e",
        "Przebieg bez ziarna mówi o tym wprost, zamiast wymyślać liczbę: None jest uczciwym "
        "zapisem tego, że wywołujący niczego nie wybrał, a zapis 42 deklarowałby konfigurację, "
        "o którą nikt nie prosił.",
    ),
    "test_run_provenance.py::every_predictor_stamps_its_run": (
        "2704debf",
        "Każdy model opisuje swój przebieg: opis jest coś wart tylko wtedy, gdy żaden model go "
        "nie omija. Prognozy trafiają na dysk w jednym miejscu, walk_forward.finish_run, które "
        "zapisuje obok nich podsumowanie z ustawieniami, a każdy model kończy pracę w run.",
    ),
    "test_run_provenance.py::no_command_offers_a_flag_that_does_nothing": (
        "93070988",
        "Żadne polecenie nie oferuje flagi, która nic nie robi: --strict i --save-metadata "
        "przyjmowało piętnaście poleceń, a nie czytało żadne. Panel oferował obie jako "
        "przełączniki.",
    ),
    "test_run_provenance.py::reading_the_configuration_does_not_touch_the_disk": (
        "a4130849",
        "Odczyt konfiguracji nie zmienia dysku: zbudowanie ustawień nie może tworzyć katalogów "
        "raw/, processed/, reports/ i meta/ pod podanym katalogiem głównym.",
    ),
    "test_run_provenance.py::the_caller_keeps_the_last_word": (
        "828fd233",
        "Ostatnie słowo należy do wywołującego: model, który wie lepiej niż nazwa przebiegu, "
        "musi móc to powiedzieć.",
    ),
    "test_run_provenance.py::the_environment_that_produced_a_run_is_recorded": (
        "ab2c7577",
        "Zapisywane jest środowisko, które wytworzyło przebieg: --save-metadata to obiecywało "
        "i niczego nie zapisywało, a teraz robi to każdy przebieg. Model wytrenowany ponownie "
        "jest porównywalny ze starym tylko wtedy, gdy zgadzają się biblioteki, które go liczą, "
        "a po fakcie nie da się tego ustalić, jeśli nie zostało zapisane.",
    ),
    "test_run_provenance.py::the_model_and_interval_come_from_the_run_slug": (
        "4ed6c861",
        "Model i interwał pochodzą z nazwy przebiegu: niezmiennik 5 zapisuje je już w nazwie "
        "katalogu, więc ponowne wyprowadzanie ich tutaj byłoby drugą definicją, która mogłaby "
        "się nie zgadzać.",
    ),
    "test_run_provenance.py::the_seed_is_recorded": ("da39a3ee", "Ziarno jest zapisywane."),
    "test_scale_free_features.py::a_narrower_dataset_is_allowed": (
        "3a3edd5e",
        "Węższy zbiór cech jest dopuszczalny: cechy przepływu zleceń wymagają kolumn, których "
        "starsze dane nie mają, a cechy śróddzienne istnieją tylko przy 1d, więc krótsza lista "
        "jest uprawniona, a nie błędna.",
    ),
    "test_scale_free_features.py::every_predictor_uses_the_include_list": (
        "273e0fac",
        "Każdy model korzysta z listy włączającej: lista wykluczająca przyjmuje każdą nową "
        "kolumnę, i tak kolumny względne trafiłyby do treningu razem ze swoimi bezwzględnymi "
        "odpowiednikami. Modele z pulą dostają kolumny z pool.load_pool, które wyznacza część "
        "wspólną training_features dla symboli, więc wystarcza bezpośrednie wywołanie albo "
        "przejście przez pulę. Zabronione jest wymienianie kolumn do pominięcia.",
    ),
    "test_scale_free_features.py::no_absolute_column_is_in_the_training_set": (
        "da39a3ee",
        "Żadna kolumna bezwzględna nie trafia do zbioru treningowego.",
    ),
    "test_scale_free_features.py::the_absolute_columns_really_do_differ": (
        "8f8fad10",
        "Kolumny bezwzględne rzeczywiście się różnią: chroni to poprzedni test przed "
        "przejściem bez treści - gdyby przeskalowanie szeregu niczego nie zmieniało, "
        "sprawdzenie niezmienniczości niczego by nie dowodziło.",
    ),
    "test_scale_free_features.py::the_duplicate_trade_count_column_is_gone": (
        "1db7f617",
        "Zduplikowana kolumna liczby transakcji została usunięta: trade_count był rzutowaniem "
        "num_trades na liczbę zmiennoprzecinkową, potrzebnym do trade_count_rel_30, a jego "
        "zapis przechowywał tę samą wielkość dwukrotnie.",
    ),
    "test_scale_free_features.py::the_order_is_stable": (
        "333e93e6",
        "Kolejność cech jest stała: modelu zapisanego przy jednej kolejności kolumn nie da się "
        "ocenić przy innej, więc kolejność nie może zależeć od iteracji po zbiorze.",
    ),
    "test_scale_free_features.py::the_training_set_is_invariant_to_the_price_level": (
        "37c63423",
        "Zbiór treningowy jest niezależny od poziomu ceny - to własność, dla której istnieje "
        "cała zmiana: dwa szeregi o identycznym kształcie, różniące się poziomem ceny "
        "tysiąckrotnie, muszą dać identyczne cechy treningowe.",
    ),
    "test_significance.py::a_coin_flip_evaluation_never_claims_to_beat_chance": (
        "da39a3ee",
        "Ocena rzutu monetą nigdy nie twierdzi, że pokonuje losowość.",
    ),
    "test_significance.py::a_coin_flip_is_not_separated_from_chance": (
        "da39a3ee",
        "Rzut monetą nie jest odróżniany od losowości.",
    ),
    "test_significance.py::a_different_seed_moves_the_interval_only_a_little": (
        "da39a3ee",
        "Inne ziarno przesuwa przedział tylko nieznacznie.",
    ),
    "test_significance.py::a_miscalibrated_forecast_has_an_interval_away_from_zero": (
        "da39a3ee",
        "Prognoza źle skalibrowana ma przedział błędu kalibracji odsunięty od zera.",
    ),
    "test_significance.py::a_refused_interval_has_no_p_value": (
        "c7fbefb9",
        "Odmówiony przedział nie ma wartości p: zbyt mało bloków i próbkowanie, które nie "
        "obejmuje własnej oceny.",
    ),
    "test_significance.py::a_resampling_that_never_crosses_the_reference_is_not_p_zero": (
        "5ae34c06",
        "Próbkowanie, które ani razu nie przekracza wartości odniesienia, nie daje p = 0: "
        "pokazuje tylko, że p leży poniżej rozdzielczości próbkowania.",
    ),
    "test_significance.py::a_smaller_p_never_gets_a_larger_q": (
        "28a08e93",
        "Mniejsze p nigdy nie dostaje większego q: bez przejścia od największej wartości w dół "
        "0,011 na pozycji 2 przeskalowałoby się do 0,022 i wyprzedziło 0,012 na pozycji 3, "
        "które skaluje się do 0,016.",
    ),
    "test_significance.py::a_well_sampled_interval_always_brackets_its_estimate": (
        "15cdc4d3",
        "Dobrze próbkowany przedział zawsze obejmuje własną ocenę: zabezpieczenie nie może "
        "działać na poprawnych danych, bo po cichu zamieniłoby każdy wynik w „nieznany”.",
    ),
    "test_significance.py::an_interval_that_cannot_bracket_its_estimate_is_refused": (
        "6a320ae3",
        "Przedział, który nie obejmuje własnej oceny, jest odrzucany. Przypadek z "
        "rzeczywistych wyników: przy 3,1 bloku dwa z pięćdziesięciu pięciu przedziałów nie "
        "obejmowały własnej oceny punktowej - 0,339 wobec [0,066; 0,335] - bo cztery "
        "wylosowane i przycięte bloki nie odtworzą szeregu, który zawiera trzy. Sama liczba "
        "bloków tego nie wychwyciła, więc niespójność wykrywana jest wprost.",
    ),
    "test_significance.py::an_unestimable_interval_never_claims_a_difference": (
        "433369c3",
        "Przedział niemożliwy do wyznaczenia nigdy nie twierdzi, że jest różnica: excludes "
        "musi zwracać None, a nie False - „nie da się stwierdzić” to nie „brak efektu”.",
    ),
    "test_significance.py::at_either_end_of_the_interval_the_p_value_is_alpha": (
        "b8b52854",
        "Na obu końcach przedziału wartość p wynosi alfa: p to odwrócony przedział "
        "95-procentowy, więc wartość odniesienia na jego końcu daje około 0,05.",
    ),
    "test_significance.py::calibration_error_is_recomputed_inside_each_resample": (
        "da84fd42",
        "Błąd kalibracji liczony jest od nowa w każdym losowaniu: ECE jest sumą ważoną po "
        "przedziałach, a nie średnią po wierszach, więc uśrednianie wartości wierszy dałoby "
        "zupełnie inną wielkość.",
    ),
    "test_significance.py::effective_samples_divides_by_the_overlap_length": (
        "da39a3ee",
        "Liczba obserwacji efektywnych dzieli liczbę wierszy przez długość nakładania się "
        "etykiet.",
    ),
    "test_significance.py::every_resample_has_the_original_length": (
        "3a5522ba",
        "Każde losowanie ma długość oryginału: krótki ostatni blok musi zostać przycięty, a "
        "nie pominięty, bo pominięcie przesuwałoby statystykę w stronę tego, co akurat "
        "zawierają wcześniejsze bloki.",
    ),
    "test_significance.py::excludes_detects_a_real_separation": (
        "da39a3ee",
        "Funkcja excludes wykrywa rzeczywiste rozdzielenie.",
    ),
    "test_significance.py::mean_interval_handles_an_unbounded_score": (
        "37067b03",
        "Przedział średniej obsługuje miarę nieograniczoną: strata logarytmiczna nie jest "
        "proporcją, więc nic nie może zakładać przedziału [0, 1].",
    ),
    "test_significance.py::no_scored_rows_is_reported_as_such": (
        "da39a3ee",
        "Brak ocenionych wierszy raportowany jest wprost.",
    ),
    "test_significance.py::on_independent_data_it_matches_the_analytic_interval": (
        "c6a8e866",
        "Na danych niezależnych wynik zgadza się z przedziałem analitycznym: przy block_size=1 "
        "nie ma czego korygować, więc nie wolno też dodawać szerokości.",
    ),
    "test_significance.py::q_values_scale_each_p_by_its_rank_and_stay_monotone": (
        "bd90664d",
        "Wartości q skalują każde p według jego pozycji i pozostają monotoniczne. Przykład "
        "liczony ręcznie: posortowane 0,005, 0,01, 0,03 i 0,04 dają 0,02, 0,02, 0,04 i 0,04 i "
        "wracają w podanej kolejności.",
    ),
    "test_significance.py::the_cli_prints_the_interval_and_the_verdict": (
        "e9384173",
        "Wiersz poleceń wypisuje przedział i werdykt: o to chodzi w tabeli - liczba podana bez "
        "nich jest błędem, któremu ten moduł ma zapobiegać.",
    ),
    "test_significance.py::the_estimate_itself_is_no_departure": (
        "da39a3ee",
        "Sama ocena punktowa nie jest odstępstwem od siebie.",
    ),
    "test_significance.py::the_evaluation_reports_an_interval_for_every_horizon": (
        "caa2bad7",
        "Ocena podaje przedział dla każdego horyzontu: sama trafność nie jest wynikiem, więc "
        "nie może dać się jej otrzymać bez przedziału.",
    ),
    "test_significance.py::the_floor_reflects_the_measured_failure": (
        "9f58268c",
        "Próg odzwierciedla zmierzony błąd: trzy bloki to próg okna treningowego zapożyczony "
        "do innego pytania, a dla stabilnego przedziału to za mało.",
    ),
    "test_significance.py::the_interval_brackets_the_point_estimate": (
        "da39a3ee",
        "Przedział obejmuje ocenę punktową.",
    ),
    "test_significance.py::the_interval_is_wider_on_correlated_data_than_on_independent_data": (
        "2fecd491",
        "Na danych skorelowanych przedział jest szerszy niż na niezależnych - to cały powód "
        "istnienia tego modułu.",
    ),
    "test_significance.py::the_p_value_agrees_with_the_interval_about_chance": (
        "af13915d",
        "Wartość p zgadza się z przedziałem co do losowości: jest mała tam, gdzie przedział "
        "wyklucza 0,5 z którejkolwiek strony, a nie tam, gdzie je obejmuje.",
    ),
    "test_significance.py::the_p_value_is_read_off_the_intervals_own_resampling": (
        "b977d2c7",
        "Wartość p odczytywana jest z próbkowania samego przedziału: to te same losowania co "
        "przedział, obok którego jest podawana, a dłuższe próbkowanie z tym samym ziarnem "
        "tylko dodaje losowania po nich.",
    ),
    "test_significance.py::the_point_estimate_is_the_statistic_itself": (
        "da39a3ee",
        "Ocena punktowa jest samą statystyką.",
    ),
    "test_significance.py::the_result_is_reproducible_from_the_seed": (
        "da39a3ee",
        "Wynik daje się odtworzyć z ziarna.",
    ),
    "test_significance.py::the_verdict_says_which_side_of_chance": (
        "63334808",
        "Werdykt mówi, po której stronie losowości leży wynik: excludes jest dwustronne - jest "
        "prawdziwe dla przebiegu, którego cały przedział leży poniżej 0,5, a tego żadne "
        "liczenie przebiegów „powyżej losowości” nie może odczytać jako umiejętności.",
    ),
    "test_significance.py::too_few_blocks_is_a_refusal_not_a_narrow_interval": (
        "a9815545",
        "Zbyt mała liczba bloków oznacza odmowę, a nie wąski przedział: horyzont roczny jest "
        "blisko tej granicy - 1249 wierszy przy nakładaniu się 364 świec to około trzech "
        "bloków, a próbkowanie jednej obserwacji raportowałoby fałszywą precyzję.",
    ),
    "test_significance.py::with_one_test_the_q_value_is_the_p_value": (
        "da39a3ee",
        "Przy jednym teście wartość q równa jest wartości p.",
    ),
    "test_task_queue.py::a_failing_task_does_not_stall_the_queue": (
        "da39a3ee",
        "Zadanie zakończone błędem nie zatrzymuje kolejki.",
    ),
    "test_task_queue.py::pending_tasks_can_be_cancelled_and_reordered": (
        "da39a3ee",
        "Zadania oczekujące można usuwać i przestawiać.",
    ),
    "test_task_queue.py::queue_endpoints_report_and_reject": (
        "da39a3ee",
        "Trasy kolejki raportują jej stan i odrzucają błędne żądania.",
    ),
    "test_task_queue.py::queue_reports_current_and_pending": (
        "da39a3ee",
        "Kolejka raportuje zadanie bieżące i oczekujące.",
    ),
    "test_task_queue.py::tasks_run_one_after_another": (
        "abd0ec21",
        "Zadania wykonywane są jedno po drugim: jeden wątek roboczy, bo etapy potoku zapisują "
        "te same partycje.",
    ),
    "test_task_queue.py::worker_restarts_after_the_queue_empties": (
        "19a4bdd9",
        "Wątek roboczy startuje ponownie po opróżnieniu kolejki: ponowne zgłoszenie po jego "
        "zakończeniu nie może doprowadzić do zakleszczenia.",
    ),
    "test_tft_batched_inference.py::a_month_is_predicted_in_one_call": (
        "da39a3ee",
        "Miesiąc prognozowany jest w jednym wywołaniu.",
    ),
    "test_tft_batched_inference.py::logits_become_a_valid_distribution": (
        "da39a3ee",
        "Logity zamieniane są na poprawny rozkład prawdopodobieństwa.",
    ),
    "test_tft_batched_inference.py::logits_survive_a_large_magnitude": (
        "e9169c9c",
        "Logity o dużej wartości nie powodują przepełnienia: softmax przesuwany jest o "
        "maksimum.",
    ),
    "test_tft_batched_inference.py::prediction_is_unwrapped_from_every_known_shape": (
        "da39a3ee",
        "Prognoza wydobywana jest z każdej znanej postaci wyniku.",
    ),
    "test_tft_batched_inference.py::target_history_is_masked_at_the_month_cutoff": (
        "f4dc026d",
        "Historia celu maskowana jest na granicy miesiąca: etykiety wymagające zamknięcia w "
        "chwili granicy albo później nie mogą trafić do kodera.",
    ),
    "test_tft_batched_inference.py::tft_end_to_end_produces_trained_predictions": (
        "6f5fbd08",
        "TFT od początku do końca daje prognozy trenowane: rzeczywisty (mały) przebieg TFT, w "
        "którym każdy horyzont się uczy, a wynik się zmienia.",
    ),
    "test_tft_batched_inference.py::the_caller_dict_survives_repeated_prediction": (
        "b7aca0c4",
        "Słownik wywołującego przetrwa wielokrotne prognozowanie. pytorch-forecasting wstawia "
        "swój kolektor do przekazanego słownika: predict wykonuje "
        "trainer_kwargs.setdefault(\"callbacks\", ... + [callback]), więc słownik używany "
        "ponownie zachowuje wywołanie zwrotne z pierwszego wywołania. Każdy kolejny Trainer "
        "zbiera wtedy do nieaktualnego obiektu, wynik wraca pusty, a horyzont po cichu staje "
        "się nietrenowany. Jeden słownik dla trzech horyzontów w miesiącu to dokładnie układ, "
        "który to wywołuje.",
    ),
    "test_tft_batched_inference.py::the_unknown_class_is_dropped_before_softmax": (
        "5380f41b",
        "Klasa nieznana jest usuwana przed softmax: NaNLabelEncoder(add_nan=True) dodaje "
        "klasę, która nie jest kierunkiem.",
    ),
    "test_tft_class_order.py::a_dataset_with_no_encoder_is_an_error_not_a_guess": (
        "da39a3ee",
        "Zbiór danych bez kodera etykiet jest błędem, a nie powodem do zgadywania.",
    ),
    "test_tft_class_order.py::a_declaration_ordered_encoder_is_the_identity": (
        "b26c6e29",
        "Koder uporządkowany według deklaracji daje odwzorowanie tożsamościowe: nic tu nie "
        "zakłada, że koder się nie zgadza - tylko że zostaje zapytany.",
    ),
    "test_tft_class_order.py::a_degenerate_row_falls_back_to_uniform": (
        "da39a3ee",
        "Zdegenerowany wiersz zastępowany jest rozkładem jednostajnym.",
    ),
    "test_tft_class_order.py::a_missing_direction_class_is_an_error": (
        "da39a3ee",
        "Brak klasy kierunku jest błędem.",
    ),
    "test_tft_class_order.py::an_encoder_without_a_nan_class_still_maps_correctly": (
        "da39a3ee",
        "Koder bez klasy NaN nadal odwzorowuje kolumny poprawnie.",
    ),
    "test_tft_class_order.py::columns_follow_the_encoder_not_the_declaration_order": (
        "da39a3ee",
        "Kolumny wynikają z kodera, a nie z kolejności deklaracji.",
    ),
    "test_tft_class_order.py::numpy_string_keys_are_accepted": (
        "fe61b512",
        "Przyjmowane są klucze w postaci napisów numpy: NaNLabelEncoder przechowuje klucze "
        "np.str_ po dopasowaniu na tablicy numpy.",
    ),
    "test_tft_class_order.py::the_encoder_really_does_sort_its_classes": (
        "ff36fa1e",
        "Koder rzeczywiście sortuje klasy - test utrwala zachowanie biblioteki, z powodu "
        "którego istnieje poprawka. Jeśli pytorch-forecasting przestanie sortować, test "
        "zawiedzie, a opis poprawki przestanie być prawdziwy - o czym warto wiedzieć od razu.",
    ),
    "test_tft_class_order.py::the_nan_column_carries_no_mass": (
        "9ba02911",
        "Kolumna NaN nie niesie masy prawdopodobieństwa: wybór po softmax i ponowna "
        "normalizacja usuwają ją dokładnie.",
    ),
    "test_tft_class_order.py::the_probability_follows_the_named_column": (
        "090d53f9",
        "Prawdopodobieństwo trafia do nazwanej kolumny: największy logit musi trafić do klasy, "
        "którą wskazuje koder.",
    ),
    "test_time_utils.py::normalize_timestamp_ms_microseconds": (
        "1f72d98a",
        "Normalizacja znacznika czasu w mikrosekundach.",
    ),
    "test_time_utils.py::normalize_timestamp_ms_milliseconds": (
        "a032ebc1",
        "Normalizacja znacznika czasu w milisekundach.",
    ),
    "test_time_utils.py::parse_date": ("af7d4f40", "Odczyt daty w formacie RRRR-MM-DD."),
    "test_time_utils.py::parse_year_month": ("f3da13df", "Odczyt daty w formacie RRRR-MM."),
    "test_time_utils.py::parse_year_month_invalid": (
        "e73a7fba",
        "Odczyt błędnej daty w formacie RRRR-MM.",
    ),
    "test_time_utils.py::timestamp_to_utc_datetime": (
        "9e16faa7",
        "Zamiana znacznika czasu na datę i godzinę UTC.",
    ),
    "test_train_window.py::a_horizon_the_history_cannot_support_is_not_quietly_widened": (
        "1468de29",
        "Horyzont, którego historia nie wystarcza, nie jest po cichu poszerzany: SOLUSDT ma "
        "2150 świec dziennych, a idealne okno roczne zajęłoby 1456 z nich i zostawiło 0,9 "
        "obserwacji ocenianej - ideał jest więc nieosiągalny i zwracany jest punkt równowagi "
        "zamiast niego.",
    ),
    "test_train_window.py::a_per_horizon_override_beats_a_shared_width": (
        "da39a3ee",
        "Nadpisanie dla horyzontu ma pierwszeństwo przed wspólną szerokością.",
    ),
    "test_train_window.py::an_explicit_width_may_be_narrower_than_the_rule_wants": (
        "2617eec6",
        "Jawnie podana szerokość może być węższa, niż wymaga reguła: odtworzenie starszego "
        "przebiegu oznacza poproszenie o to, czego ten przebieg używał, także gdy wiadomo już, "
        "że to za mało.",
    ),
    "test_train_window.py::an_explicit_width_overrides_the_rule": (
        "da39a3ee",
        "Jawnie podana szerokość zastępuje regułę.",
    ),
    "test_train_window.py::bounds_pair_each_horizon_with_its_own_span": (
        "9f38be53",
        "Granice łączą każdy horyzont z jego własną rozpiętością: okno i rozpiętość w przód "
        "idą w parze, a połączenie okna jednego horyzontu z rozpiętością innego uczyłoby na "
        "etykietach, których model nie mógł znać.",
    ),
    "test_train_window.py::each_horizon_gets_its_own_width": (
        "da39a3ee",
        "Każdy horyzont dostaje własną szerokość okna.",
    ),
    "test_train_window.py::every_horizon_clears_both_floors_on_the_history_we_have": (
        "f39bca82",
        "Każdy horyzont przekracza oba progi na dostępnej historii - to cel całej zmiany. Na "
        "dziewięciu latach BTCUSDT wszystkie cztery horyzonty są mierzalne po obu stronach, "
        "czego stałe okno 730-dniowe dla roku w ogóle nie zapewniało.",
    ),
    "test_train_window.py::every_predictor_reports_its_power": (
        "76cef554",
        "Każdy model raportuje swoją moc statystyczną. resolve_train_windows celowo nie "
        "ostrzega, bo użyteczne ostrzeżenie wymaga zakresu ocenianego, znanego dopiero po "
        "wczytaniu danych. Ten test istnieje, ponieważ ostrzeżenia przez miesiące brakowało, a "
        "horyzont roczny podawano z dokładnością do trzech miejsc po przecinku na podstawie "
        "jednej obserwacji. Wszystkie trzy wywołania żyją teraz w walk_forward.plan_run, a "
        "model może dotrzeć do pętli kroczącej wyłącznie z planem - bezpośrednio albo przez "
        "moduł wczytujący modeli z pulą.",
    ),
    "test_train_window.py::long_horizons_are_sized_by_the_observations_statistics_needs": (
        "bd662bf7",
        "Długie horyzonty mają okno wyznaczane liczbą obserwacji potrzebną statystyce: przy "
        "364 świecach próg wierszy jest spełniony długo przed progiem obserwacji, więc "
        "rozstrzyga ten drugi.",
    ),
    "test_train_window.py::more_history_is_what_actually_helps": (
        "9d272796",
        "Pomaga dopiero dłuższa historia - to uczciwy wniosek z poprzedniego testu: horyzont "
        "niemierzalny naprawia więcej danych, a nie lepsze ustawienie.",
    ),
    "test_train_window.py::no_predictor_slices_a_training_set_by_hand": (
        "4f2e525e",
        "Żaden model nie wycina zbioru treningowego ręcznie: windows.bounds jest jedyną drogą "
        "do wycinka treningowego. Bezpośrednie wywołanie training_bounds(...) znów "
        "przyjmowałoby okno jako luźny argument, a tak właśnie szerokość jednego horyzontu "
        "trafiłaby w parę z rozpiętością innego.",
    ),
    "test_train_window.py::short_horizons_are_sized_by_the_rows_a_model_needs": (
        "ab581536",
        "Krótkie horyzonty mają okno wyznaczane liczbą wierszy potrzebną modelowi: przy "
        "horyzoncie 7 świec niemal całe okno ma etykiety, więc obserwacji jest pod dostatkiem, "
        "a wiąże posiadanie dość wierszy, by w ogóle dopasować model.",
    ),
    "test_train_window.py::the_balance_point_is_the_best_any_window_could_do": (
        "8841a4c9",
        "Punkt równowagi jest najlepszym, co może dać jakiekolwiek okno, a nie dowolnym "
        "rozwiązaniem zastępczym: połowa historii maksymalizuje słabszą ze stron, więc żadna "
        "inna szerokość też nie uratowałaby horyzontu.",
    ),
    "test_train_window.py::the_predictors_take_the_flag": (
        "da39a3ee",
        "Modele przyjmują flagę szerokości okna.",
    ),
    "test_train_window.py::the_two_floors_swap_over_between_the_horizons": (
        "4f93d057",
        "Oba progi zamieniają się rolami między horyzontami: żaden z nich osobno nie wystarczy "
        "- jeden głodzi długie horyzonty z obserwacji, drugi krótkie z wierszy.",
    ),
    "test_train_window.py::the_widest_window_covers_every_horizon": (
        "481e16a0",
        "Najszersze okno obejmuje każdy horyzont: normalizacja dopasowywana jest raz na świecę "
        "i współdzielona, więc jej zasięg musi sięgać początku najgłębszego wycinka "
        "treningowego.",
    ),
    "test_train_window.py::the_window_scales_with_the_interval": (
        "d5250e39",
        "Okno skaluje się z interwałem: horyzonty są rozpiętościami kalendarzowymi, więc "
        "przebieg 1h musi oznaczać ten sam rzeczywisty czas (niezmiennik 2), a okno wyrażone w "
        "świecach musi za tym nadążać.",
    ),
    "test_unlabelled_bars.py::a_bar_without_a_label_does_not_stop_training": (
        "da39a3ee",
        "Świeca bez etykiety nie przerywa uczenia.",
    ),
    "test_versioned_predictions.py::another_runs_artifacts_are_left_alone": (
        "da39a3ee",
        "Artefakty innego przebiegu pozostają nienaruszone.",
    ),
    "test_versioned_predictions.py::delete_ai_model_not_found": (
        "da39a3ee",
        "Usuwanie nieistniejącego przebiegu zgłasza jego brak.",
    ),
    "test_versioned_predictions.py::delete_ai_model_removes_run_folder": (
        "da39a3ee",
        "Usuwanie przebiegu usuwa jego katalog.",
    ),
    "test_versioned_predictions.py::deleting_a_run_removes_its_derived_artifacts": (
        "67c91bb9",
        "Usunięcie przebiegu usuwa jego artefakty pochodne (niezmiennik 5): artefakt wskazuje "
        "przebieg źródłowy, więc nie może go przeżyć. Pozostawione na dysku symulacja i "
        "rekomendacje usuniętego przebiegu byłyby raportowane przez każdy program "
        "przeglądający te katalogi, obok bieżących.",
    ),
    "test_versioned_predictions.py::deleting_a_run_without_derived_artifacts_still_works": (
        "da39a3ee",
        "Usunięcie przebiegu bez artefaktów pochodnych nadal działa.",
    ),
    "test_versioned_predictions.py::evaluate_model_predictions_builds_safe_command": (
        "da39a3ee",
        "Ocena prognoz modelu buduje bezpieczne polecenie.",
    ),
    "test_versioned_predictions.py::get_ai_models_parses_interval": (
        "da39a3ee",
        "Lista przebiegów odczytuje interwał z nazwy.",
    ),
    "test_versioned_predictions.py::get_ai_predictions_versioned_run": (
        "da39a3ee",
        "Prognozy zwracane są dla wersjonowanego przebiegu.",
    ),
    "test_versioned_predictions.py::get_feature_importances_returns_json": (
        "da39a3ee",
        "Ważność cech zwracana jest jako JSON.",
    ),
    "test_versioned_predictions.py::parse_versioned_prediction_folder_embedded_model_type": (
        "da39a3ee",
        "Odczyt nazwy katalogu z rodzajem modelu zapisanym w nazwie.",
    ),
    "test_versioned_predictions.py::parse_versioned_prediction_folder_legacy_format": (
        "da39a3ee",
        "Odczyt nazwy katalogu w starszym formacie.",
    ),
    "test_versioned_predictions.py::parse_versioned_prediction_folder_new_format": (
        "da39a3ee",
        "Odczyt nazwy katalogu w nowym formacie.",
    ),
    "test_versioned_predictions.py::run_training_includes_interval_in_command": (
        "da39a3ee",
        "Uruchomienie treningu przekazuje interwał w poleceniu.",
    ),
    "test_versioned_predictions.py::versioned_prediction_folder_name_with_interval": (
        "da39a3ee",
        "Nazwa wersjonowanego katalogu prognoz zawiera interwał.",
    ),
    "test_walk_forward.py::a_refused_horizon_is_written_untrained_and_counted": (
        "da39a3ee",
        "Odrzucony horyzont zapisywany jest jako nietrenowany i liczony.",
    ),
    "test_walk_forward.py::a_run_is_written_with_its_summary": (
        "da39a3ee",
        "Przebieg zapisywany jest razem z podsumowaniem.",
    ),
    "test_walk_forward.py::an_empty_range_is_refused_rather_than_written": (
        "da39a3ee",
        "Pusty zakres jest odrzucany, a nie zapisywany.",
    ),
    "test_walk_forward.py::an_unselected_horizon_is_never_asked": (
        "da39a3ee",
        "Model nigdy nie jest pytany o horyzont, którego nie wybrano.",
    ),
    "test_walk_forward.py::empirical_depth_counts_each_side_and_falls_back_to_uniform": (
        "da39a3ee",
        "Empiryczny rozkład głębokości liczy każdą stronę osobno, a bez danych wraca do "
        "rozkładu jednostajnego.",
    ),
    "test_walk_forward.py::one_row_per_bar_in_range_in_order": (
        "da39a3ee",
        "Jeden wiersz na każdą świecę zakresu, w kolejności.",
    ),
    "test_walk_forward.py::seeding_only_happens_when_asked": (
        "da39a3ee",
        "Ziarno ustawiane jest tylko na żądanie.",
    ),
    "test_walk_forward.py::selecting_horizons_does_not_change_the_windows": (
        "1c8def2f",
        "Wybór horyzontów nie zmienia okien: --horizons decyduje, co jest dopasowywane, nigdy "
        "- na czym dopasowywane są pozostałe.",
    ),
    "test_walk_forward.py::the_model_refits_once_per_month_at_its_first_bar": (
        "da39a3ee",
        "Model douczany jest raz w miesiącu, na jego pierwszej świecy.",
    ),
    "test_window_seed.py::both_deep_predictors_reseed_per_window": (
        "045979fb",
        "Oba modele głębokie ustawiają ziarno od nowa w każdym oknie: funkcja pomocnicza, "
        "której nikt nie wywołuje, niczego nie naprawia. Ponowne ziarnowanie żyje w jednym "
        "miejscu, SequenceForecaster.reseed, a obie sieci są prognostami sekwencyjnymi i "
        "wywołują je tuż przed zbudowaniem modelu każdego okna.",
    ),
    "test_window_seed.py::every_field_changes_the_seed": (
        "da39a3ee",
        "Każde pole tożsamości okna zmienia ziarno.",
    ),
    "test_window_seed.py::no_collisions_across_a_decade_of_windows": (
        "da39a3ee",
        "Brak kolizji ziaren w oknach z dziesięciu lat.",
    ),
    "test_window_seed.py::seeds_are_valid_for_torch": (
        "da39a3ee",
        "Ziarna są poprawne dla biblioteki torch.",
    ),
    "test_window_seed.py::selecting_fewer_horizons_leaves_the_selected_ones_untouched": (
        "e1a4eb72",
        "Wybór mniejszej liczby horyzontów nie zmienia wybranych: --horizons month musi "
        "wytrenować ten sam model miesięczny co pełny przebieg. Na tej własności opiera się "
        "test replikacji: TFT kosztuje około 29 minut na horyzont, więc sprawdzenie jednej "
        "hipotezy o horyzoncie miesięcznym na czterech nowych symbolach jest wykonalne tylko "
        "wtedy, gdy trening tego horyzontu osobno daje ten sam model. Obie potrzebne własności "
        "wynikają z budowy systemu, a nie z przypadku. Zmierzone na XGBoost dla ETHUSDT "
        "2025-01 - 2026-04: wynik miesięczny różni się na 0 z 485 świec między przebiegiem "
        "czterohoryzontowym a --horizons month.",
    ),
    "test_window_seed.py::the_model_name_is_hashed_not_summed": (
        "948faca1",
        "Nazwa modelu jest haszowana, a nie sumowana: pierwsza wersja dodawała kody znaków, a "
        "'tft' i 'gru' sumują się do 334, więc oba modele dzieliły każde ziarno w projekcie.",
    ),
    "test_window_seed.py::the_seed_ignores_which_other_horizons_ran": (
        "c3ac9267",
        "Ziarno nie zależy od tego, jakie inne horyzonty trenowano - to arytmetyka poprzedniej "
        "własności: ziarno okna wskazuje tylko własny horyzont, więc przebieg trenujący jeden "
        "horyzont ustawia ziarno dokładnie tak jak przebieg trenujący cztery.",
    ),
    "test_window_seed.py::the_seed_is_stable_across_processes": (
        "911c77ff",
        "Ziarno jest stałe między procesami: hash() jest solony osobno w każdym procesie, więc "
        "nie może być tu użyty - przebieg nie odtworzyłby się następnego dnia. Test utrwala "
        "konkretne wartości.",
    ),
    "test_window_seed.py::the_symbol_is_the_only_thing_that_changed": (
        "78808b3d",
        "Symbol jest jedyną zmianą: jego dodanie nie może naruszyć izolacji, dla której "
        "istnieje ta funkcja - dwa przebiegi różniące się jednym hiperparametrem nadal "
        "zgadzają się wszędzie, gdzie ten hiperparametr nie sięga.",
    ),
    "test_window_seed.py::the_symbol_reaches_the_seed_from_both_deep_predictors": (
        "7a79f4c4",
        "Symbol trafia do ziarna z obu modeli głębokich: ziarno zróżnicowane, którego nikt nie "
        "przekazuje, nie różnicuje niczego.",
    ),
    "test_window_seed.py::two_symbols_do_not_share_an_initialisation": (
        "0c992517",
        "Dwa symbole nie dzielą inicjalizacji - to pominięcie kosztowało jeden wynik. "
        "window_seed haszowało (ziarno bazowe, model, horyzont, rok, miesiąc) i na tym "
        "kończyło, więc w danym oknie douczania każdy symbol losował te same wagi, te same "
        "maski dropoutu i tę samą kolejność porcji. Cztery symbole uczone pod jednym ziarnem "
        "bazowym nie były więc czterema niezależnymi losowaniami „tego modelu na nowych "
        "danych”, lecz jednym losowaniem sekwencji inicjalizacji zastosowanym do czterech "
        "zbiorów, a replikację na czterech symbolach odczytano jako niezależny dowód. Wspólny "
        "czynnik nie jest mały: ponowne ziarnowanie przesunęło miesięczną trafność jednego "
        "symbolu średnio o 0,049, a najwyżej o 0,064, wobec deklarowanego efektu 0,043.",
    ),
}

# Commands, by their full path.
COMMANDS: dict[str, tuple[str, str]] = {
    "prosper aggregate": ("aadda1c0", "Agreguje świece z interwału 1m do 1h, 1d i 1w."),
    "prosper backfill": ("2857825b", "Pobiera miesięczne archiwa z Binance Public Data."),
    "prosper eval backtest": (
        "ba5bdb92",
        "Uruchamia symulację handlową na jednym wersjonowanym przebiegu prognozy.",
    ),
    "prosper eval batch": (
        "d0d71717",
        "Ocenia hurtowo dostępne przebiegi modeli; nadaje się do uruchamiania z harmonogramu "
        "zadań.",
    ),
    "prosper eval compare": (
        "12a2614f",
        "Zestawia wszystkie ocenione przebiegi w jednym rankingu.",
    ),
    "prosper eval predictions": (
        "6eb01ed7",
        "Ocenia wersjonowany przebieg prognozy i zapisuje artefakty jakości oraz rekomendacji.",
    ),
    "prosper eval stability": (
        "4db84a77",
        "Śledzi jakość prognozy jednego przebiegu miesiąc po miesiącu, aby sprawdzić, czy się "
        "utrzymuje.",
    ),
    "prosper features build": (
        "046e6fcc",
        "Wyznacza cechy techniczne i zapisuje je do pliku parquet.",
    ),
    "prosper labels build": (
        "7b558426",
        "Zapisuje etykiety kierunku i głębokości wszystkich horyzontów dla jednego interwału.",
    ),
    "prosper planner windows": (
        "03099db1",
        "Wyznacza okna działania i rekomendacje (od Strong Buy do Strong Sell).",
    ),
    "prosper predict baseline": (
        "dd88f343",
        "Wyznacza prognozy probabilistyczne modelu odniesienia dla każdego horyzontu.",
    ),
    "prosper predict gru": (
        "4d0a3cff",
        "Wyznacza prognozy probabilistyczne sieci GRU dla każdego horyzontu.",
    ),
    "prosper predict ml": (
        "130ecd78",
        "Wyznacza prognozy probabilistyczne wzmacniania gradientowego scikit-learn dla każdego "
        "horyzontu.",
    ),
    "prosper predict tft": (
        "0e0efdf2",
        "Wyznacza prognozy probabilistyczne modelu TFT (Temporal Fusion Transformer).",
    ),
    "prosper predict xgboost": (
        "b4de1c7e",
        "Wyznacza prognozy probabilistyczne modelu XGBoost dla każdego horyzontu.",
    ),
    "prosper qa check": (
        "5d6bf162",
        "Uruchamia kontrole jakości danych (sortowanie, duplikaty, luki, niezmienniki OHLC).",
    ),
    "prosper symbols list": (
        "099a0477",
        "Pobiera listę symboli rynku SPOT z Binance exchangeInfo i zapisuje ją do "
        "data/meta/binance_spot_symbols.json.",
    ),
    "prosper ui": ("75831ee5", "Uruchamia panel przeglądarkowy Prosper."),
}

# HTTP routes, as methods and path.
ROUTES: dict[str, tuple[str, str]] = {
    "DELETE /api/ai/models/{symbol}/{model_type}/{timestamp}": (
        "3bc15c8a",
        "Usuwa przebieg prognozy wraz ze wszystkimi artefaktami pochodnymi.",
    ),
    "DELETE /api/data/{symbol}": (
        "3bc15c8a",
        "Usuwa wszystkie dane jednego symbolu z jeziora danych.",
    ),
    "DELETE /api/task/queue/{task_id}": (
        "f97e56dd",
        "Usuwa zadanie oczekujące; zadanie już wykonywane nie jest przerywane.",
    ),
    "GET /api/ai/evaluations": (
        "3bc15c8a",
        "Lista ocenionych przebiegów z miarami jakości, z filtrem symbolu i modelu oraz "
        "sortowaniem.",
    ),
    "GET /api/ai/evaluations/status": (
        "3bc15c8a",
        "Stan ostatniej oceny wsadowej: liczba ocenionych i nieudanych przebiegów oraz "
        "ostrzeżenia.",
    ),
    "GET /api/ai/evaluations/{symbol}/{model_type}/{timestamp}": (
        "01ff5c86",
        "Podsumowanie, kalibracja i najgorsze wiersze jednego ocenionego przebiegu.",
    ),
    "GET /api/ai/evaluations/{symbol}/{model_type}/{timestamp}/flags": (
        "53b5f80c",
        "Flagi jakości występujące w przebiegu wraz z ich liczbą, do filtra w panelu.",
    ),
    "GET /api/ai/feature_importances": (
        "3bc15c8a",
        "Ważność cech zapisana przez jeden przebieg modelu.",
    ),
    "GET /api/ai/models": ("3bc15c8a", "Inwentarz zapisanych przebiegów prognozy."),
    "GET /api/ai/predictions": (
        "ba2bc41a",
        "Prognozy przebiegu, opcjonalnie zawężone do jednego miesiąca kalendarzowego.",
    ),
    "GET /api/ai/predictions/months": (
        "afb9d860",
        "Posortowana lista miesięcy (RRRR-MM) obecnych w przebiegu.",
    ),
    "GET /api/backtest/{symbol}": (
        "040ecf09",
        "Symuluje handel na jednym przebiegu prognozy i zwraca przebieg kapitału oraz "
        "statystyki.",
    ),
    "GET /api/data/dates": (
        "3bc15c8a",
        "Daty, którymi panel wypełnia formularz pobierania: dzisiejsza, wczorajsza i "
        "najwcześniejszy miesiąc.",
    ),
    "GET /api/data/inventory": (
        "3bc15c8a",
        "Inwentarz jeziora danych: symbole, zakresy dat, interwały i zajętość dysku.",
    ),
    "GET /api/data/klines/{symbol}/{interval}": (
        "3bc15c8a",
        "Świece jednego symbolu i interwału do wykresu, z limitem liczby i zakresem dat.",
    ),
    "GET /api/meta/backtest-defaults": (
        "db71a214",
        "Wartości domyślne, którymi panel wypełnia formularz symulacji.",
    ),
    "GET /api/meta/symbols": (
        "3bc15c8a",
        "Lista symboli rynku SPOT wraz z walutami bazowymi i kwotowanymi.",
    ),
    "GET /api/meta/training-defaults": (
        "e98ece60",
        "Wartości domyślne, którymi panel wypełnia formularz treningu.",
    ),
    "GET /api/runs": ("3ed99b9d", "Lista wersjonowanych przebiegów prognozy, od najnowszego."),
    "GET /api/stability/{symbol}": (
        "459d82d3",
        "Jakość jednego przebiegu jako szereg czasowy, w postaci gotowej do wykresu.",
    ),
    "GET /api/status": ("3bc15c8a", "Stan serwera."),
    "GET /api/task/logs": (
        "3bc15c8a",
        "Dziennik bieżącego zadania od wskazanego wiersza, wraz ze stanem kolejki.",
    ),
    "GET /api/task/queue": (
        "0a5bb7fd",
        "Kolejka zadań w wersji serwera; panel wyświetla ją zamiast własnej kopii.",
    ),
    "GET /api/task/status": ("3bc15c8a", "Stan kolejki zadań i zadania wykonywanego."),
    "POST /api/eval/batch": ("1b588d4e", "Zgłasza do kolejki hurtową ocenę przebiegów modeli."),
    "POST /api/eval/run": ("1e4c9d12", "Zgłasza do kolejki ocenę przebiegu prognozy."),
    "POST /api/pipeline/aggregate": (
        "4ffa3621",
        "Zgłasza agregację świec 1m do wybranych interwałów oraz przebudowę cech i etykiet.",
    ),
    "POST /api/pipeline/repair": (
        "27a8f657",
        "Naprawia dane symbolu; kroki naprawy serwer wyprowadza z bieżącego inwentarza.",
    ),
    "POST /api/pipeline/run": (
        "891f73f1",
        "Zgłasza potok danych: pobranie, agregację, cechy i etykiety.",
    ),
    "POST /api/task/queue/{task_id}/move": (
        "e13bfe72",
        "Przesuwa zadanie oczekujące o jedno miejsce wcześniej (-1) albo później (+1).",
    ),
    "POST /api/train/run": ("08f85c5e", "Zgłasza do kolejki trening modelu."),
}

# Configuration fields, by name.
SETTINGS: dict[str, tuple[str, str]] = {
    "api_rate_limit_backoff_base": (
        "3bc15c8a",
        "Podstawa wykładniczego odstępu między ponowieniami zapytań po przekroczeniu limitu "
        "interfejsu giełdy.",
    ),
    "baseline_rolling_window_days": (
        "3bc15c8a",
        "Stała szerokość okna modelu odniesienia w dniach; bez wartości każdy horyzont "
        "wyprowadza własne okno.",
    ),
    "binance_api_base": ("3bc15c8a", "Adres interfejsu REST giełdy Binance."),
    "binance_public_data_base": (
        "3bc15c8a",
        "Adres serwisu Binance Public Data z archiwami historycznymi.",
    ),
    "data_root": ("f80be4cc", "Katalog główny wszystkich danych."),
    "deterministic": ("09a8154c", "Włącza deterministyczne uczenie (tryb badawczy)."),
    "download_retry_backoff": (
        "3bc15c8a",
        "Podstawa wykładniczego odstępu między ponowieniami pobierania.",
    ),
    "download_retry_max": ("3bc15c8a", "Największa liczba ponowień pobierania jednego archiwum."),
    "download_timeout": ("3bc15c8a", "Limit czasu pobierania jednego archiwum, w sekundach."),
    "download_workers": ("3bc15c8a", "Liczba równoległych wątków pobierania."),
    "label_depth_bins_default": (
        "3bc15c8a",
        "Domyślne granice kubełków głębokości ruchu, w procentach.",
    ),
    "log_format": ("4d817750", "Format wpisu w dzienniku."),
    "log_level": ("50d692b9", "Poziom szczegółowości dziennika."),
    "meta_dir": (
        "3bc15c8a",
        "Katalog metadanych, np. listy symboli; domyślnie podkatalog meta katalogu głównego.",
    ),
    "planner_round_trip_cost": (
        "c18fe143",
        "Koszt pełnej transakcji (kupna i sprzedaży), który sygnał musi pokryć, zanim planer "
        "zareaguje.",
    ),
    "processed_data_dir": (
        "3bc15c8a",
        "Katalog danych przetworzonych; domyślnie podkatalog processed katalogu głównego.",
    ),
    "qa_gap_tolerance_seconds": (
        "3bc15c8a",
        "Tolerancja luki czasowej między świecami w kontroli jakości, w sekundach.",
    ),
    "raw_data_dir": (
        "3bc15c8a",
        "Katalog surowych archiwów; domyślnie podkatalog raw katalogu głównego.",
    ),
    "reports_dir": (
        "3bc15c8a",
        "Katalog raportów i przebiegów; domyślnie podkatalog reports katalogu głównego.",
    ),
    "seed": (
        "ef49c853",
        "Ziarno generatora liczb losowych zapewniające odtwarzalność (tryb badawczy).",
    ),
    "trading_fee_rate": ("8832a8ff", "Stawka prowizji od wartości transakcji w symulacji."),
    "trading_slippage_proxy_rate": (
        "37cbdcbd",
        "Przybliżenie poślizgu cenowego od wartości transakcji w symulacji.",
    ),
}

# Flag descriptions, keyed by the English help text.
OPTIONS: dict[str, str] = {
    "Additional target intervals (used only for syntax: --to 1h 1d 1w).": (
        "Dodatkowe interwały docelowe (tylko dla zapisu --to 1h 1d 1w)."
    ),
    "Bar interval (1d, 1h, ...)": "Interwał świec (1d, 1h, ...)",
    "Bars of trailing change the no-model trend benchmark reads": (
        "Liczba świec wstecz, z których reguła momentum (punkt odniesienia bez modelu) "
        "odczytuje trend"
    ),
    "Base interval for features (MVP: 1d)": "Interwał bazowy cech (obecnie 1d)",
    "Base interval for labels": "Interwał bazowy etykiet",
    "Comma-separated horizons to train (default: all). Unlisted ones are written as untrained "
    "and skipped downstream.": (
        "Horyzonty do treningu rozdzielone przecinkami (domyślnie wszystkie). Pominięte "
        "zapisywane są jako nietrenowane i pomijane w dalszych etapach."
    ),
    "Depth bin definitions": "Definicje kubełków głębokości",
    "Dry run mode (simulate only)": "Tryb próbny: tylko symulacja, bez zapisu",
    "Enable deterministic training": "Włącza deterministyczne uczenie",
    "Encoder sequence length": "Długość sekwencji kodera",
    "End date in YYYY-MM format": "Data końcowa w formacie RRRR-MM",
    "End date in YYYY-MM-DD": "Data końcowa w formacie RRRR-MM-DD",
    "End date in YYYY-MM-DD format": "Data końcowa w formacie RRRR-MM-DD",
    "Epoch budget per window; early stopping usually ends training sooner": (
        "Limit epok na okno; wczesne zatrzymanie zwykle kończy uczenie wcześniej"
    ),
    "Exchange fee per trade, as a fraction (default from settings)": (
        "Prowizja giełdy od transakcji jako ułamek (domyślnie z konfiguracji)"
    ),
    "Expected move a signal must clear to be acted on, as a fraction": (
        "Oczekiwany ruch, który sygnał musi przekroczyć, aby zostać wykonany, jako ułamek"
    ),
    "Extra symbols to pool into the training set, e.g. 'BTCUSDT,SOLUSDT'. Predictions are "
    "still for --symbol alone.": (
        "Dodatkowe symbole dołączane do zbioru treningowego, np. 'BTCUSDT,SOLUSDT'. Prognoza "
        "nadal dotyczy wyłącznie symbolu z --symbol."
    ),
    "Feature interval (1m, 1h, 1d)": "Interwał cech (1m, 1h, 1d)",
    "Filter by quote asset (e.g., USDT)": "Filtr waluty kwotowanej (np. USDT)",
    "GRU hidden layer size": "Rozmiar warstwy ukrytej GRU",
    "Horizon driving the signal": "Horyzont, z którego pochodzi sygnał",
    "Host IP to bind to": "Adres IP, na którym nasłuchuje serwer",
    "Initial Capital in USDT": "Kapitał początkowy w USDT",
    "Local data lake root": "Katalog główny lokalnego jeziora danych",
    "Local data lake root directory": "Katalog główny lokalnego jeziora danych",
    "Maximum runs to evaluate": "Największa liczba przebiegów do oceny",
    "Model type": "Rodzaj modelu",
    "Month (1-12)": "Miesiąc (1-12)",
    "Number of parallel workers": "Liczba równoległych wątków pobierania",
    "Open the dashboard on startup": "Otwiera panel w przeglądarce po uruchomieniu",
    "Optional interval filter": "Opcjonalny filtr interwału",
    "Optional model filter": "Opcjonalny filtr modelu",
    "Optional symbol filter": "Opcjonalny filtr symbolu",
    "Output JSON file path": "Ścieżka wyjściowego pliku JSON",
    "Per-horizon epoch ceilings, e.g. 'short=20,medium=5'. Overrides the budget.": (
        "Limity epok osobno dla horyzontów, np. 'short=20,medium=5'. Zastępują limit ogólny."
    ),
    "Per-horizon training windows in days, e.g. 'year=1456'. Overrides the derived width.": (
        "Okna treningowe w dniach osobno dla horyzontów, np. 'year=1456'. Zastępuje szerokość "
        "wyprowadzoną."
    ),
    "Port to run the UI server on": "Port serwera panelu",
    "Prediction/data interval": "Interwał prognozy i danych",
    "Random seed for reproducibility": (
        "Ziarno generatora liczb losowych, zapewniające odtwarzalność"
    ),
    "Reprocess months that already have Parquet (needed after a schema change)": (
        "Przetwarza ponownie miesiące, które mają już plik Parquet (potrzebne po zmianie "
        "schematu)"
    ),
    "Rolling window in calendar days, one width for every horizon. Omit it and each horizon "
    "derives its own from its span and the history available.": (
        "Okno kroczące w dniach kalendarzowych, jedna szerokość dla wszystkich horyzontów. Bez "
        "flagi każdy horyzont wyprowadza własne okno z rozpiętości i dostępnej historii."
    ),
    "Sequence length for GRU input": "Długość sekwencji wejściowej GRU",
    "Slippage proxy per trade, as a fraction": (
        "Przybliżenie poślizgu cenowego na transakcję jako ułamek"
    ),
    "Smallest exposure change worth trading, as a fraction of equity": (
        "Najmniejsza zmiana ekspozycji warta transakcji, jako ułamek kapitału"
    ),
    "Source interval": "Interwał źródłowy",
    "Source model; latest run if omitted": "Model źródłowy; bez flagi najnowszy przebieg",
    "Source run interval": "Interwał przebiegu źródłowego",
    "Source run timestamp; latest run if omitted": (
        "Znacznik czasu przebiegu źródłowego; bez flagi najnowszy przebieg"
    ),
    "Start date in YYYY-MM format": "Data początkowa w formacie RRRR-MM",
    "Start date in YYYY-MM-DD": "Data początkowa w formacie RRRR-MM-DD",
    "Start date in YYYY-MM-DD format": "Data początkowa w formacie RRRR-MM-DD",
    "TFT hidden layer size": "Rozmiar warstwy ukrytej TFT",
    "Target intervals, comma-separated (e.g. 1h,1d,1w)": (
        "Interwały docelowe rozdzielone przecinkami (np. 1h,1d,1w)"
    ),
    "Target shares per grade, e.g. 'strong_buy=1.0,buy=0.5'. Hold cannot be set.": (
        "Docelowy udział w kapitale dla każdego stopnia rekomendacji, np. "
        "'strong_buy=1.0,buy=0.5'. Wartości Hold nie można ustawić."
    ),
    "Time interval (1m, 1h, 1d)": "Interwał czasowy (1m, 1h, 1d)",
    "Trading symbol": "Symbol instrumentu",
    "Trading symbol (e.g., BTCUSDT)": "Symbol instrumentu (np. BTCUSDT)",
    "Training window in calendar days, one width for every horizon. Omit it and each horizon "
    "derives its own from its span and the history available.": (
        "Okno treningowe w dniach kalendarzowych, jedna szerokość dla wszystkich horyzontów. "
        "Bez flagi każdy horyzont wyprowadza własne okno z rozpiętości i dostępnej historii."
    ),
    "Versioned prediction timestamp": "Znacznik czasu wersjonowanego przebiegu prognozy",
    "Write the table as CSV": "Zapisuje tabelę w formacie CSV",
    "XGBoost learning rate": "Współczynnik uczenia XGBoost",
    "XGBoost max_depth": "Maksymalna głębokość drzewa XGBoost (max_depth)",
    "XGBoost n_estimators": "Liczba drzew XGBoost (n_estimators)",
    "Year": "Rok",
}
