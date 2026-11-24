$ErrorActionPreference = 'Stop'
$recordRoot = $PSScriptRoot
$sourceSpecs = @(
    @('lesniewski_ruskai.pdf', 'https://arxiv.org/pdf/math-ph/9808016'),
    @('grasselli_streater.pdf', 'https://math.mcmaster.ca/~grasselli/GrasselliStreater01.pdf'),
    @('fawzi_saunderson_parrilo.pdf', 'https://arxiv.org/pdf/1705.00812'),
    @('bresar_cabrera_fosner_villena.pdf', 'https://www.impan.pl/shop/en/publication/transaction/download/product/90340'),
    @('ohara_ishi_tsuchiya.html', 'https://link.springer.com/article/10.1007/s41884-023-00116-x'),
    @('nagaoka_fujiwara.html', 'https://arxiv.org/html/2307.03431v3'),
    @('ciaglia_dinocera.pdf', 'https://arxiv.org/pdf/2206.10394'),
    @('li_wang_lee_arora.pdf', 'https://papers.nips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Paper-Conference.pdf')
)
$records = foreach ($spec in $sourceSpecs) {
    $targetPath = Join-Path (Join-Path $recordRoot 'sources') $spec[0]
    $row = [ordered]@{file=('sources/' + $spec[0]); url=$spec[1]; retrieved_at_utc=[DateTime]::UtcNow.ToString('o')}
    try {
        Invoke-WebRequest -Uri $spec[1] -OutFile $targetPath -TimeoutSec 15 -UseBasicParsing
        $row.status = 'saved'
        $row.bytes = (Get-Item -LiteralPath $targetPath).Length
        $row.sha256 = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } catch {
        $row.status = 'failed'
        $row.error = $_.Exception.Message
    }
    $row
}
$outputRecord = [ordered]@{
    purpose='Primary source byte provenance; relevant passages inspected independently via web tools.'
    source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    records=$records
    preceding_failure='The python command resolved to an inaccessible Microsoft WindowsApps alias; no Python source retrieval ran.'
}
$outputRecord | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $recordRoot 'SOURCE_HASHES.json') -Encoding utf8
$records | Select-Object file, status
