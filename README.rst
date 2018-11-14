peer_fixed_effect
========================

Python package of ``Arcidiacono, P., Foster, G., Goodpaster, N., & Kinsler, J. (2012). Estimating spillovers using panel data, with an application to the classroom. Quantitative Economics, 3(3), 421-470.``


Install
---------------

::

    python setup.py install


If you want to learn more about setup.py files, check out this repository.

Uninstall
---------------

::

    python setup.py install --record files.txt
    cat files.txt | xargs rm -rf
    rm files.txt


Example
---------------

::

    import pandas as pd
    from peer_fixed_effect import PeerFixedEffectRegression
    df = pd.read_csv('hogehoge.csv')
    pfer = PeerFixedEffectRegression()
    pfer.fit(
        x=df[['xit0', 'xit1', 'xit2']].values,
        y=df['yit'].values,
        group=df['group'].values,
        ids=df['ids'].values,
        times=df['time'].values,
    )


✨🍰✨
