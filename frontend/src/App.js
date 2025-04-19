import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  Chip,
  Stack,
  AppBar,
  Toolbar,
  ThemeProvider,
  createTheme,
  alpha,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  DialogContentText,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import PersonIcon from '@mui/icons-material/Person';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import BusinessIcon from '@mui/icons-material/Business';
import DescriptionIcon from '@mui/icons-material/Description';
import ShareIcon from '@mui/icons-material/Share';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import BookmarkBorderIcon from '@mui/icons-material/BookmarkBorder';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import SchoolIcon from '@mui/icons-material/School';
import axios from 'axios';

// Create a modern theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 500,
    },
    h3: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      fontWeight: 600,
      letterSpacing: '-0.5px',
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          transition: 'transform 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-4px)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 16,
            backgroundColor: 'white',
            transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
            },
            '&.Mui-focused': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.12)',
            },
          },
          '& .MuiOutlinedInput-notchedOutline': {
            borderWidth: '1.5px',
            borderColor: 'rgba(25, 118, 210, 0.1)',
          },
          '& .MuiInputLabel-root': {
            fontSize: '1.1rem',
            fontWeight: 500,
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          textTransform: 'none',
          fontWeight: 600,
          fontSize: '1rem',
          padding: '12px 24px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 20px rgba(0, 0, 0, 0.15)',
          },
        },
      },
    },
  },
});

// Add this at the top of the file
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function PatentCard({ patent, onBookmark, onShare, onClick, isBookmarked }) {
  return (
    <Card onClick={onClick} sx={{ cursor: 'pointer', mb: 2 }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box flex={1}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
              {patent.title}
            </Typography>
            <Typography color="textSecondary" gutterBottom>
              Patent Number: {patent.patent_number}
            </Typography>
            {patent.similarity_score && (
              <Chip
                label={`Relevance: ${(patent.similarity_score * 100).toFixed(1)}%`}
                color="primary"
                size="small"
                sx={{ mb: 1 }}
              />
            )}
          </Box>
          <Box>
            <Tooltip title={isBookmarked ? "Remove Bookmark" : "Add Bookmark"}>
              <IconButton onClick={(e) => { e.stopPropagation(); onBookmark(patent.patent_number); }}>
                {isBookmarked ? <BookmarkIcon /> : <BookmarkBorderIcon />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Share">
              <IconButton onClick={(e) => { e.stopPropagation(); onShare(patent); }}>
                <ShareIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        <Typography variant="body2" paragraph sx={{ color: 'text.secondary' }}>
          {patent.abstract}
        </Typography>
        {patent.predicted_classes && patent.predicted_classes.length > 0 && (
          <Box mb={2}>
            <Typography variant="subtitle2" gutterBottom>
              Predicted Technology Areas:
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {patent.predicted_classes.map((cls, index) => (
                <Chip
                  key={index}
                  label={`${cls.description} (${(cls.probability * 100).toFixed(1)}%)`}
                  color="info"
                  variant="outlined"
                  size="small"
                  sx={{ mb: 1 }}
                />
              ))}
            </Stack>
          </Box>
        )}
        <Stack direction="row" spacing={2} flexWrap="wrap">
          {patent.filing_date && (
            <Chip
              icon={<CalendarTodayIcon />}
              label={`Filed: ${new Date(patent.filing_date).toLocaleDateString()}`}
              variant="outlined"
              color="primary"
              size="small"
            />
          )}
          {patent.inventors && patent.inventors.length > 0 && (
            <Chip
              icon={<PersonIcon />}
              label={`Inventors: ${patent.inventors.join(', ')}`}
              variant="outlined"
              color="secondary"
              size="small"
            />
          )}
          {patent.assignee && (
            <Chip
              icon={<BusinessIcon />}
              label={`Assignee: ${patent.assignee}`}
              variant="outlined"
              color="default"
              size="small"
            />
          )}
          {patent.patent_type && (
            <Chip
              icon={<DescriptionIcon />}
              label={`Type: ${patent.patent_type}`}
              variant="outlined"
              color="default"
              size="small"
            />
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [patents, setPatents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [bookmarkedPatents, setBookmarkedPatents] = useState(new Set());
  const [selectedPatent, setSelectedPatent] = useState(null);
  const [showPatentDialog, setShowPatentDialog] = useState(false);
  const [searchMode, setSearchMode] = useState('semantic'); // 'semantic' or 'keyword'
  const [placeholderIndex, setPlaceholderIndex] = useState(0);
  const [relevantPatents, setRelevantPatents] = useState([]);
  const [irrelevantPatents, setIrrelevantPatents] = useState([]);
  const [modelStats, setModelStats] = useState(null);

  const searchPlaceholders = [
    "Try: 'A device that helps cars navigate using satellites'",
    "Try: 'Method for secure online payment using fingerprints'",
    "Try: 'Technology for making phones charge wirelessly'",
    "Try: 'System for recognizing faces in security cameras'",
    "Try: 'Artificial intelligence for predicting weather'",
  ];

  // Rotate placeholder text every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setPlaceholderIndex((prev) => (prev + 1) % searchPlaceholders.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Fetch model stats when component mounts
    fetchModelStats();
  }, []);

  const fetchModelStats = async () => {
    try {
      const response = await fetch(`${API_URL}/model-stats`);
      const data = await response.json();
      setModelStats(data);
    } catch (error) {
      console.error('Error fetching model stats:', error);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_URL}/search`, {
        query: searchQuery,
        max_results: 10,
        search_mode: searchMode
      });
      
      setPatents(response.data);
    } catch (err) {
      let errorMessage = 'Error searching patents. Please try again.';
      if (err.response) {
        errorMessage = `Error: ${err.response.data.detail || err.response.statusText}`;
      } else if (err.request) {
        errorMessage = 'No response received from the server. Please check if the backend is running.';
      } else {
        errorMessage = `Error: ${err.message}`;
      }
      setError(errorMessage);
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleBookmark = (patentNumber) => {
    setBookmarkedPatents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(patentNumber)) {
        newSet.delete(patentNumber);
      } else {
        newSet.add(patentNumber);
      }
      return newSet;
    });
  };

  const handleShare = (patent) => {
    const shareData = {
      title: patent.title,
      text: `Check out this patent: ${patent.title}\nPatent Number: ${patent.patent_number}`,
      url: `https://patents.google.com/patent/${patent.patent_number}`
    };
    navigator.share(shareData).catch(console.error);
  };

  const handlePatentClick = (patent) => {
    setSelectedPatent(patent);
    setShowPatentDialog(true);
  };

  const handleRelevanceFeedback = (patent, isRelevant) => {
    if (isRelevant) {
      setRelevantPatents([...relevantPatents, patent]);
      setIrrelevantPatents(irrelevantPatents.filter(p => p.patent_number !== patent.patent_number));
    } else {
      setIrrelevantPatents([...irrelevantPatents, patent]);
      setRelevantPatents(relevantPatents.filter(p => p.patent_number !== patent.patent_number));
    }
  };

  const submitFeedback = async () => {
    if (!searchQuery || (relevantPatents.length === 0 && irrelevantPatents.length === 0)) {
      return;
    }

    try {
      const response = await fetch(`${API_URL}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          relevant_patents: relevantPatents,
          irrelevant_patents: irrelevantPatents
        }),
      });

      if (response.ok) {
        setRelevantPatents([]);
        setIrrelevantPatents([]);
        fetchModelStats();
        alert('Thank you for your feedback!');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ flexGrow: 1, minHeight: '100vh', bgcolor: 'background.default' }}>
        <AppBar position="static" elevation={0} sx={{ bgcolor: 'white', color: 'text.primary' }}>
          <Toolbar>
            <Box display="flex" alignItems="center">
              <img 
                src="/logo.png" 
                alt="PatentAI Logo" 
                style={{ 
                  height: 60,
                  objectFit: 'contain'
                }} 
              />
            </Box>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ py: 4 }}>
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Typography 
              variant="h3" 
              component="h1" 
              gutterBottom 
              sx={{ 
                fontWeight: 600,
                letterSpacing: '-0.5px',
                background: 'linear-gradient(45deg, #1976d2, #304FFE)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 2,
                fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
              }}
            >
              PatentAI - AI Patent Search Tool
            </Typography>
            <Typography 
              variant="subtitle1" 
              color="text.secondary" 
              sx={{ 
                mb: 4,
                fontSize: '1.1rem',
                fontWeight: 400,
                opacity: 0.87
              }}
            >
              Search and analyze patents using advanced AI technology
            </Typography>
          </Box>
          
          <Paper 
            elevation={0} 
            sx={{ 
              p: 4, 
              mb: 4, 
              borderRadius: 4,
              bgcolor: 'white',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
              border: '1.5px solid rgba(25, 118, 210, 0.08)',
              transition: 'transform 0.2s ease-in-out',
              '&:hover': {
                transform: 'translateY(-4px)',
              },
            }}
          >
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Box display="flex" justifyContent="center" mb={2}>
                  <Button
                    variant={searchMode === 'semantic' ? 'contained' : 'outlined'}
                    onClick={() => setSearchMode('semantic')}
                    sx={{ mr: 1, borderRadius: 8 }}
                  >
                    AI-Powered Search
                  </Button>
                  <Button
                    variant={searchMode === 'keyword' ? 'contained' : 'outlined'}
                    onClick={() => setSearchMode('keyword')}
                    sx={{ borderRadius: 8 }}
                  >
                    Keyword Search
                  </Button>
                </Box>
                <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 2 }}>
                  {searchMode === 'semantic' 
                    ? "AI search understands concepts and finds related patents even with different wording"
                    : "Keyword search looks for exact word matches in patent titles and abstracts"}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={9}>
                <TextField
                  fullWidth
                  label="Search Patents"
                  variant="outlined"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder={searchPlaceholders[placeholderIndex]}
                  InputProps={{
                    sx: { 
                      py: 0.5,
                      fontSize: '1.1rem',
                      '&::placeholder': {
                        opacity: 0.7,
                        fontStyle: 'italic',
                      },
                    },
                    startAdornment: (
                      <SearchIcon 
                        sx={{ 
                          mr: 1, 
                          color: 'primary.main',
                          opacity: 0.7
                        }} 
                      />
                    ),
                  }}
                />
              </Grid>
              <Grid item xs={12} sm={3}>
                <Button
                  fullWidth
                  variant="contained"
                  color="primary"
                  onClick={handleSearch}
                  disabled={loading}
                  size="large"
                  sx={{
                    height: '100%',
                    minHeight: '56px',
                  }}
                >
                  Search Patents
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {error && (
            <Typography color="error" align="center" sx={{ mb: 2 }}>
              {error}
            </Typography>
          )}

          {loading ? (
            <Box display="flex" justifyContent="center" my={4}>
              <CircularProgress />
            </Box>
          ) : (
            <Grid container spacing={3}>
              {patents.map((patent) => (
                <Grid item xs={12} key={patent.patent_number}>
                  <PatentCard
                    patent={patent}
                    onBookmark={handleBookmark}
                    onShare={handleShare}
                    onClick={() => handlePatentClick(patent)}
                    isBookmarked={bookmarkedPatents.has(patent.patent_number)}
                  />
                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <IconButton
                      color={relevantPatents.some(p => p.patent_number === patent.patent_number) ? 'success' : 'default'}
                      onClick={(e) => { e.stopPropagation(); handleRelevanceFeedback(patent, true); }}
                    >
                      <ThumbUpIcon />
                    </IconButton>
                    <IconButton
                      color={irrelevantPatents.some(p => p.patent_number === patent.patent_number) ? 'error' : 'default'}
                      onClick={(e) => { e.stopPropagation(); handleRelevanceFeedback(patent, false); }}
                    >
                      <ThumbDownIcon />
                    </IconButton>
                  </Box>
                </Grid>
              ))}
            </Grid>
          )}

          {/* Model Stats */}
          {modelStats && (
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                Model Statistics
              </Typography>
              <Typography>Model: {modelStats.model_name}</Typography>
              <Typography>Training Data Size: {modelStats.training_data_size}</Typography>
              <Typography>Validation Data Size: {modelStats.validation_data_size}</Typography>
            </Paper>
          )}

          {/* Feedback Submit Button */}
          {(relevantPatents.length > 0 || irrelevantPatents.length > 0) && (
            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={submitFeedback}
                startIcon={<SchoolIcon />}
              >
                Submit Feedback
              </Button>
            </Box>
          )}
        </Container>

        <Dialog
          open={showPatentDialog}
          onClose={() => setShowPatentDialog(false)}
          maxWidth="md"
          fullWidth
        >
          {selectedPatent && (
            <>
              <DialogTitle>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="h6">{selectedPatent.title}</Typography>
                  <Box>
                    <Tooltip title={bookmarkedPatents.has(selectedPatent.patent_number) ? "Remove Bookmark" : "Add Bookmark"}>
                      <IconButton onClick={(e) => { e.stopPropagation(); handleBookmark(selectedPatent.patent_number); }}>
                        {bookmarkedPatents.has(selectedPatent.patent_number) ? <BookmarkIcon /> : <BookmarkBorderIcon />}
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Share">
                      <IconButton onClick={(e) => { e.stopPropagation(); handleShare(selectedPatent); }}>
                        <ShareIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
              </DialogTitle>
              <DialogContent>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" color="textSecondary">
                    Patent Number: {selectedPatent.patent_number}
                  </Typography>
                </Box>
                <Typography variant="body1" paragraph>
                  {selectedPatent.abstract}
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="textSecondary">Inventors</Typography>
                    <Typography variant="body1">
                      {selectedPatent.inventors ? selectedPatent.inventors.join(', ') : 'N/A'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="textSecondary">Assignee</Typography>
                    <Typography variant="body1">
                      {selectedPatent.assignee || 'N/A'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="textSecondary">Filing Date</Typography>
                    <Typography variant="body1">
                      {selectedPatent.filing_date ? new Date(selectedPatent.filing_date).toLocaleDateString() : 'N/A'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="textSecondary">Patent Type</Typography>
                    <Typography variant="body1">
                      {selectedPatent.patent_type || 'N/A'}
                    </Typography>
                  </Grid>
                </Grid>
              </DialogContent>
              <DialogActions>
                <Button onClick={() => setShowPatentDialog(false)}>Close</Button>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => window.open(`https://patents.google.com/patent/US${selectedPatent.patent_number}`, '_blank')}
                >
                  View Full Patent
                </Button>
              </DialogActions>
            </>
          )}
        </Dialog>
      </Box>
    </ThemeProvider>
  );
}

export default App; 