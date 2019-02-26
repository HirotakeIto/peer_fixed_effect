import pandas as pd
from peer_fixed_effect.teacher_effect.structure.simple import SimpleTeacherEffectMixin
# df = pd.read_excel('tests/testdata/kinsler.xlsx', sheets='data')
# df = pd.read_excel('tests/testdata/kinsler.xlsx', sheets='data')
# TeacherEffectMixin.get_teacher_effect_discounted_cumsum_except_now(df, df, id_n, time_n, eft_jt_n, sigma)

def test_data():
    id_col = 'ids'
    tid_col = 'tid'
    grade_col = 'grade'
    time_col = 'time'
    y_col = 'y'
    eft_i_col = 'effect_i'
    eft_it_col = 'effect_it'
    eft_jt_col = 'effect_tid_t'
    # setup test data
    import numpy as np
    import pandas as pd
    n_id = 5000
    n_tid = 30
    n_time = 3
    grade_range = list(range(1, 7))
    start_time_range = np.array(grade_range)[np.argsort(grade_range)[0:(len(grade_range) - n_time + 1)]]
    persistence = 0.2
    sigma = persistence
    var_tid = 1 # iid
    var_id = 1 # iid
    ids = np.kron(np.arange(n_id).reshape((n_id, 1)), np.ones(shape=(n_time, 1)))
    grade_id = (
        np.kron(
            np.random.choice(start_time_range, size=(n_id, 1), replace=True),
            np.ones(shape=(n_time, 1))
        ) +
        np.kron(
            np.ones(shape=(n_id, 1)),
            np.array(range(0,n_time)).reshape((n_time, 1))
        )
    )
    effect_i = np.kron(
        np.random.normal(0, scale=var_id, size=(n_id, 1)),
        np.ones(shape=(n_time, 1))
    )
    effect_it = effect_i
    noise_it = np.kron(
        np.random.normal(0, scale=var_id, size=(n_id, 1))*1,
        np.ones(shape=(n_time, 1))
    )
    id_tid = np.random.choice(np.arange(n_tid), size=(n_id*n_time), replace=True)
    df_id = pd.DataFrame(
        np.c_[ids, grade_id, id_tid, effect_i, effect_it, noise_it],
        columns=['ids', 'grade', 'tid', 'effect_i', 'effect_it', 'noise_it']
    )
    tids = np.kron(np.arange(n_tid).reshape((n_tid, 1)), np.ones(shape=(len(grade_range), 1)))
    grades_tid = np.kron(
        np.ones(shape=(n_tid, 1)),
        np.array(grade_range).reshape((len(grade_range), 1))
    )
    effect_tid = np.random.normal(0, scale=var_tid, size=(n_tid*len(grade_range), 1))
    df_tid = pd.DataFrame(
        np.c_[tids, grades_tid, effect_tid],
        columns=['tid', 'grade', 'effect_tid_t'])

    df = pd.merge(df_id, df_tid, on=['tid', 'grade']).sort_values(['ids', 'grade']).reset_index(drop=True)
    df['max_grade'] = df.groupby(['ids'])['grade'].transform('max').sort_index()
    df['cumsum_effect_tid_t'] = SimpleTeacherEffectMixin().get_teacher_effect_ijgt_discounted_cumsum(
        df,
        id_col='ids',
        grade_col='grade',
        eft_jt_col='effect_tid_t',
        max_grade_col= 'max_grade',
        sigma=persistence)
    df['y'] = df['effect_it'] + df['cumsum_effect_tid_t'] + df['noise_it']
    df.to_csv('test.csv', index=False)
    print(df[['cumsum_effect_tid_t', 'effect_tid_t']])
    print('sigma=0の時は回らないことに注意')
