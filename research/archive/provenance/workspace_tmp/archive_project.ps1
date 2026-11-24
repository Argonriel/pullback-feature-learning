$ErrorActionPreference = 'Stop'
$sourceRepo = 'C:\Users\liwei\Documents\ChatGPT\277\pullback-feature-learning'
$archiveRoot = 'C:\Users\liwei\Documents\Research\pullback-feature-learning'
$workspaceRoot = 'C:\Users\liwei\Documents\ChatGPT\277'
if (Test-Path -LiteralPath $archiveRoot) { throw 'The archive destination already exists; refusing to overwrite it.' }
New-Item -ItemType Directory -Path $archiveRoot | Out-Null
$archiveEntries = [System.Collections.Generic.List[object]]::new()

function Copy-VerifiedFile([string]$Source, [string]$RelativeDestination) {
    $destination = [System.IO.Path]::GetFullPath((Join-Path $archiveRoot $RelativeDestination))
    if (-not $destination.StartsWith($archiveRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'A destination escaped the intended archive directory.'
    }
    $parentDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parentDirectory -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $destination
    $sourceHash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash
    $destinationHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    if ($sourceHash -ne $destinationHash) { throw "Archive verification failed: $RelativeDestination" }
    $archiveEntries.Add([pscustomobject]@{
        source = $Source
        archive_path = $RelativeDestination.Replace('\', '/')
        bytes = (Get-Item -LiteralPath $Source).Length
        sha256 = $sourceHash.ToLowerInvariant()
    })
}

foreach ($folder in @('docs', 'reports', 'runs', 'configs')) {
    $sourceFolder = Join-Path $sourceRepo $folder
    foreach ($item in Get-ChildItem -LiteralPath $sourceFolder -Recurse -File) {
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'Unexpected source link.' }
        $relative = [System.IO.Path]::GetRelativePath($sourceRepo, $item.FullName)
        Copy-VerifiedFile $item.FullName $relative
    }
}
Copy-VerifiedFile (Join-Path $sourceRepo 'README.md') 'PROJECT_OVERVIEW.md'
Copy-VerifiedFile (Join-Path $workspaceRoot '277final_proposal_draft.tex') 'provenance\277final_proposal_draft.tex'
foreach ($item in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'tmp') -File) {
    Copy-VerifiedFile $item.FullName ('provenance\workspace_tmp\' + $item.Name)
}

$manifest = [pscustomobject]@{
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    project = 'Pullback Feature Learning'
    source_repository = $sourceRepo
    source_commit = '2e43a2a423d17e7e2237add4229c01bd40427788'
    archive_root = $archiveRoot
    verified_files = $archiveEntries.Count
    files = $archiveEntries.ToArray()
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $archiveRoot 'provenance\archive_manifest.json') -Encoding utf8

$index = @'
# Pullback Feature Learning: local research archive

This folder is the primary local home for project analysis, research writing, experiment data, plots, and reproducibility records. Repository code, configurations, and tests remain in the separate source checkout. All project materials are in English.

## Start here

- [Project plan and proposal alignment](docs/project_plan.md)
- [Original proposal](docs/proposal.tex)
- [Project overview before the repository cleanup](PROJECT_OVERVIEW.md)
- [Mathematical derivation](docs/mathematics.md)
- [Literature review](docs/literature.md) and [references](docs/references.bib)
- [Formal experiment interpretation](docs/formal_findings.md)
- [Long-training interpretation](docs/long_training_findings.md)
- [Frozen-feature probe findings](docs/probe_findings.md)
- [Representation and displacement findings](docs/representation_findings.md)

## Data and figures

`reports/` preserves the verified report archives, raw data, metrics, checkpoint weights, audits, SVG figures, and result summaries. `runs/` preserves the original local execution outputs, including PNG previews. `configs/` is the configuration snapshot at the time of this move. The layout preserves relative links between research documents and report artifacts.

- [Formal study](reports/formal-v1/README.md)
- [Long-training diagnostic](reports/long-training-v1/README.md)
- [Frozen-feature probes](reports/probes-v1/README.md)
- [Feature clouds and spectra](reports/representation-analysis-v1/README.md)
- [Refined displacement comparison](reports/displacement-refinement-v1/README.md)

## Preservation and future work

The archive was copied and checked file by file using SHA-256 before any repository files were removed. [The manifest](provenance/archive_manifest.json) records original locations, archived paths, sizes, and checksums. `provenance/workspace_tmp/` includes the independent audit scripts and a verified Git bundle containing the pre-cleanup repository history.

Future analysis notes, report drafts, and generated outputs belong here. The source checkout may contain ignored directory links named `docs`, `reports`, and `runs` pointing into this folder so existing experiment commands keep working. Those links and their contents are not included in new Git commits.

This cleanup removes research materials from the current GitHub file tree. It does not erase historical Git commits. The local Git bundle preserves the original research provenance independently of the remote repository.
'@
$index | Set-Content -LiteralPath (Join-Path $archiveRoot 'README.md') -Encoding utf8
Write-Output ("Verified {0} archived files at {1}" -f $archiveEntries.Count, $archiveRoot)
